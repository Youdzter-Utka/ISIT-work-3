"""Offline inference on downloaded HF models; intended for Google Colab."""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["military", "civilian"]
PROMPT = (
    "Classify the largest, primary aircraft in this photo. "
    "Use the main intended role of its aircraft variant, not only its operator. "
    "Military includes fighters, military trainers, transports, tankers and patrol aircraft. "
    "Civilian includes passenger, civilian cargo, private, sport, agricultural and firefighting aircraft. "
    "Reply with exactly one word: military or civilian."
)


def read_manifest():
    """Validate the frozen 30-image dataset before inference or evaluation."""
    import csv

    with (ROOT / "data/manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
        raise ValueError("The dataset must contain 30 unique IDs.")
    if any(sum(row["label"] == label for row in rows) != 15 for label in LABELS):
        raise ValueError("Expected 15 military and 15 civilian images.")
    hashes = []
    for row in rows:
        if row["reviewed"] != "1" or row["hard"] not in ("0", "1"):
            raise ValueError("Unreviewed or invalid annotation: " + row["id"])
        path = (ROOT / row["image_path"]).resolve()
        if not path.is_relative_to((ROOT / "data/images").resolve()):
            raise ValueError("Image path outside the dataset.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Image changed: " + row["id"])
        hashes.append(digest)
    if len(set(hashes)) != 30:
        raise ValueError("Duplicate photos found.")
    return rows


def load_config():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    models = config["models"]
    if len(models) != 2 or len({model["id"] for model in models}) != 2:
        raise ValueError("Exactly two different models are required.")
    for model in models:
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("Pin each model to a full Hub commit SHA.")
    if config["max_new_tokens"] < 1 or config["max_image_edge"] < 1:
        raise ValueError("Invalid generation settings.")
    return config


def experiment_id(config, rows, runtime):
    """Include model revisions, image hashes, prompt and inference environment."""
    payload = {
        "config": config,
        "prompt": PROMPT,
        "data": [
            {key: row[key] for key in ("id", "sha256", "label", "hard")}
            for row in rows
        ],
        "runtime": runtime,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_label(raw):
    """Accept a single class name with optional final period or outer quotes."""
    label = raw.strip().lower().strip('"\'').removesuffix(".").strip()
    return label if label in LABELS else "invalid"


def read_log(path):
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt log line {line_number}; restore a backup.") from exc
        records.append(record)
    keys = [(record["model"], record["id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate predictions in log.")
    return records


def runtime_info(torch):
    packages = {}
    for package in ("torch", "transformers", "accelerate", "Pillow", "huggingface_hub"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "python": platform.python_version(),
        "packages": packages,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "dtype": "float16" if device == "cuda" else "float32",
        "attention": "sdpa",
    }


def _load_model(spec, device, dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # token=False prevents use of any cached HF credentials. Both repos are public.
    kwargs = {"revision": spec["revision"], "token": False, "trust_remote_code": False}
    processor_kwargs = {}
    if spec["family"] == "qwen2_5_vl":
        processor_kwargs = {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
    processor = AutoProcessor.from_pretrained(spec["id"], **kwargs, **processor_kwargs)
    if spec["family"] == "smolvlm":
        # Documented SmolVLM image processor option limits image splitting.
        processor.image_processor.size = {"longest_edge": 1024}
    model = AutoModelForImageTextToText.from_pretrained(
        spec["id"], **kwargs, torch_dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    return processor, model


def _synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def _predict(processor, model, row, config, torch, device, dtype):
    from PIL import Image

    with Image.open(ROOT / row["image_path"]) as source:
        image = source.convert("RGB")
    image.thumbnail((config["max_image_edge"], config["max_image_edge"]))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT},
    ]}]
    # Only photo pixels and PROMPT enter the model. No filename, URL or truth label.
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    _synchronize(torch, device)
    start = time.perf_counter()
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    # Cast only floating tensors; token IDs and grid dimensions must remain integers.
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            inputs[key] = value.to(dtype=dtype)
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
            use_cache=True,
        )
    _synchronize(torch, device)
    latency = time.perf_counter() - start
    # Decode ONLY new tokens; otherwise the prompt itself would contaminate parsing.
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    generated_tokens = int(new_tokens.shape[1])
    return raw, latency, generated_tokens


def run(limit=None, allow_cpu=False):
    """Load one model at a time; save each answer and resume without repeating it."""
    import torch

    rows = read_manifest()
    config = load_config()
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer.")
    runtime = runtime_info(torch)
    device = runtime["device"]
    if device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "GPU не выделен. В Colab: Среда выполнения → Сменить среду выполнения → T4 GPU. "
            "Если бесплатная квота недоступна, попробуйте позже. "
            "Для осознанного медленного запуска на CPU: run(allow_cpu=True)."
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(config["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    signature = experiment_id(config, rows, runtime)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    log_path = results / "predictions.jsonl"
    records = read_log(log_path)
    if any(record["experiment"] != signature for record in records):
        raise ValueError("Settings, data or runtime changed. Archive previous results first.")
    done = {(record["model"], record["id"]) for record in records}
    expected_keys = {(spec["id"], row["id"]) for spec in config["models"] for row in rows}
    if not done <= expected_keys:
        raise ValueError("Unexpected model or image in saved predictions.")
    session = {"experiment": signature, "config": config, "prompt": PROMPT, "runtime": runtime}
    session_path = results / "experiment.json"
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    (results / "status.json").write_text(json.dumps({
        "status": "complete" if done == expected_keys else "running",
        "predictions_obtained": len(done), "planned_predictions": 60,
    }, indent=2), encoding="utf-8")
    count = 0
    for spec in config["models"]:
        pending = [row for row in rows if (spec["id"], row["id"]) not in done]
        if not pending:
            continue
        if limit is not None and count >= limit:
            break
        print("Загрузка модели:", spec["id"], flush=True)
        processor = model = None
        try:
            load_start = time.perf_counter()
            processor, model = _load_model(spec, device, dtype)
            load_seconds = time.perf_counter() - load_start
            for row in pending:
                if limit is not None and count >= limit:
                    break
                raw, seconds, token_count = _predict(
                    processor, model, row, config, torch, device, dtype,
                )
                prediction = parse_label(raw)
                record = {
                    "experiment": signature, "id": row["id"], "model": spec["id"],
                    "revision": spec["revision"], "truth": row["label"],
                    "hard": int(row["hard"]), "sha256": row["sha256"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw": raw, "prediction": prediction,
                    "status": "invalid" if prediction == "invalid" else "ok",
                    "latency_s": seconds, "generated_tokens": token_count,
                    "model_load_s": load_seconds, "device": device,
                }
                # Each line is flushed on close; only a complete response is recorded.
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.add((spec["id"], row["id"]))
                count += 1
                print(f"{len(done):02}/60 · {row['id']} · {spec['id']} → {prediction}", flush=True)
        finally:
            del processor, model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    complete = done == expected_keys
    status = {"status": "complete" if complete else "incomplete",
              "predictions_obtained": len(done), "planned_predictions": 60}
    (results / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("Готово: 60/60." if complete else f"Сохранено {len(done)}/60. Повторите run() для продолжения.")
    return status


def report():
    """Evaluate only a complete, consistent set of 60 real saved predictions."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from math import comb
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    rows = read_manifest()
    config = load_config()
    results = ROOT / "results"
    records = read_log(results / "predictions.jsonl")
    expected = {(spec["id"], row["id"]) for spec in config["models"] for row in rows}
    if {(record["model"], record["id"]) for record in records} != expected:
        raise ValueError(f"Для сравнения нужны все 60 ответов; сейчас {len(records)}.")
    session = json.loads((results / "experiment.json").read_text(encoding="utf-8"))
    signature = experiment_id(config, rows, session["runtime"])
    if session["experiment"] != signature or any(r["experiment"] != signature for r in records):
        raise ValueError("Predictions belong to a different experiment.")
    annotation = {row["id"]: row for row in rows}
    revisions = {spec["id"]: spec["revision"] for spec in config["models"]}
    for record in records:
        row = annotation[record["id"]]
        if (record["truth"] != row["label"] or record["sha256"] != row["sha256"]
                or record["hard"] != int(row["hard"])
                or record["revision"] != revisions[record["model"]]
                or record["prediction"] != parse_label(record["raw"])):
            raise ValueError("Saved response does not match frozen data or parsing rules.")
    df = pd.DataFrame(records)
    metrics, per_class = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, spec in zip(axes, config["models"]):
        model_id = spec["id"]
        group = df[df.model == model_id]
        precision, recall, f1, support = precision_recall_fscore_support(
            group.truth, group.prediction, labels=LABELS, zero_division=0,
        )
        metrics.append({
            "model": model_id, "n": len(group),
            "accuracy": accuracy_score(group.truth, group.prediction),
            "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()), "coverage": float(group.prediction.isin(LABELS).mean()),
            "hard_accuracy": float((group[group.hard == 1].truth == group[group.hard == 1].prediction).mean()),
            "mean_latency_s": group.latency_s.mean(), "median_latency_s": group.latency_s.median(),
        })
        for i, label in enumerate(LABELS):
            per_class.append({"model": model_id, "class": label, "precision": precision[i],
                              "recall": recall[i], "f1": f1[i], "support": int(support[i])})
        cm = confusion_matrix(group.truth, group.prediction, labels=LABELS + ["invalid"])[:2, :]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), LABELS + ["invalid"])
        ax.set_yticks(range(2), LABELS)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Truth")
        ax.set_title(model_id.split("/")[-1])
        for y in range(2):
            for x in range(3):
                ax.text(x, y, str(cm[y, x]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    table = pd.DataFrame(metrics).sort_values("f1_macro", ascending=False)
    table.to_csv(results / "metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(results / "per_class_metrics.csv", index=False)
    details = df.merge(pd.DataFrame(rows)[["id", "page_title", "reason", "file_url", "image_path"]], on="id")
    details.to_csv(results / "all_predictions.csv", index=False)
    errors = details[details.truth != details.prediction]
    errors.to_csv(results / "errors.csv", index=False)
    a, b = [spec["id"] for spec in config["models"]]
    pivot = df.pivot(index="id", columns="model", values="prediction")
    truth = df[df.model == a].set_index("id").truth
    correct_a, correct_b = pivot[a].eq(truth), pivot[b].eq(truth)
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    n = n10 + n01
    pvalue = min(1.0, 2 * sum(comb(n, k) for k in range(min(n10, n01) + 1)) / 2**n) if n else 1.0
    summary = {"status": "complete", "n_images": 30, "n_predictions": 60,
               "model_a": a, "model_b": b, "a_only_correct": n10,
               "b_only_correct": n01, "mcnemar_exact_p": pvalue}
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    return table, errors, summary
