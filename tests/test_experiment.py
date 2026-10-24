"""Unit tests for inference control and metric calculation."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


import matplotlib.pyplot
import pandas
import sklearn.metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experiment as e  # noqa: E402


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = e.ROOT
        cls.rows = e.read_manifest()
        cls.config = e.load_config()
        cls.real_load_model = staticmethod(e._load_model)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text(
            json.dumps(self.config), encoding="utf-8"
        )
        self.patches = [
            patch.object(e, "ROOT", self.root),
            patch.object(e, "read_manifest", return_value=self.rows),
            patch.object(
                e,
                "runtime_info",
                return_value={
                    "device": "cuda",
                    "device_name": "mock-only",
                    "dtype": "float16",
                },
            ),
            patch.object(e, "_load_model", return_value=(object(), object())),
        ]
        self.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda _: None,
            empty_cache=lambda: None,
        )
        self.fake_torch = types.SimpleNamespace(
            cuda=self.cuda,
            float16="float16",
            float32="float32",
            manual_seed=lambda _: None,
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(benchmark=False)
            ),
        )
        self.patches.append(
            patch.dict(sys.modules, {"torch": self.fake_torch})
        )
        for item in self.patches:
            item.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def perfect(self, processor, model, row, config, torch, device, dtype):
        return row["label"], 0.01, 2

    def test_dataset_has_30_balanced_reviewed_unique_images(self):
        self.assertEqual(len(self.rows), 30)
        self.assertEqual(sum(r["label"] == "military" for r in self.rows), 15)
        self.assertEqual(sum(r["hard"] == "1" for r in self.rows), 20)
        self.assertEqual(len({r["sha256"] for r in self.rows}), 30)

    def test_parser_does_not_guess_class_from_explanation(self):
        for text in ["military", "MILITARY.", ' "military" ']:
            self.assertEqual(e.parse_label(text), "military")
        for text in [
            "",
            "The aircraft is military",
            "military or civilian",
            '{"label":"military"}',
        ]:
            self.assertEqual(e.parse_label(text), "invalid")

    def test_limit_resume_and_no_repeated_generation(self):
        with patch.object(
            e, "_predict", side_effect=self.perfect
        ) as prediction:
            self.assertEqual(e.run(limit=2)["predictions_obtained"], 2)
            self.assertEqual(prediction.call_count, 2)
            self.assertEqual(e.run()["status"], "complete")
            self.assertEqual(prediction.call_count, 60)
            self.assertEqual(e.run()["status"], "complete")
            self.assertEqual(prediction.call_count, 60)
        self.assertEqual(
            len(e.read_log(self.root / "results/predictions.jsonl")), 60
        )

    def test_generation_failure_keeps_previous_answers(self):
        with patch.object(
            e,
            "_predict",
            side_effect=[("military", 0.01, 2), RuntimeError("mock OOM")],
        ):
            with self.assertRaisesRegex(RuntimeError, "mock OOM"):
                e.run()
        self.assertEqual(
            len(e.read_log(self.root / "results/predictions.jsonl")), 1
        )

    def test_changed_configuration_rejects_resume(self):
        with patch.object(e, "_predict", side_effect=self.perfect):
            e.run(limit=1)
        config = dict(self.config, seed=99)
        (self.root / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            e.run()

    def test_incomplete_results_cannot_claim_comparison(self):
        with patch.object(e, "_predict", side_effect=self.perfect):
            e.run(limit=2)
        with self.assertRaisesRegex(ValueError, "60"):
            e.report()

    def test_metrics_perfect_and_invalid_counted_as_error(self):
        with patch.object(e, "_predict", side_effect=self.perfect):
            e.run()
        metrics, errors, summary = e.report()
        self.assertTrue(all(metrics.accuracy == 1))
        self.assertTrue(all(metrics.f1_macro == 1))
        self.assertTrue(errors.empty)
        self.assertEqual(summary["mcnemar_exact_p"], 1)
        path = self.root / "results/predictions.jsonl"
        records = e.read_log(path)
        records[0].update(
            raw="unparseable", prediction="invalid", status="invalid"
        )
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        metrics, errors, _ = e.report()
        self.assertAlmostEqual(metrics.accuracy.min(), 29 / 30)
        self.assertAlmostEqual(metrics.coverage.min(), 29 / 30)
        self.assertEqual(len(errors), 1)

    def test_tampered_truth_is_rejected(self):
        with patch.object(e, "_predict", side_effect=self.perfect):
            e.run()
        path = self.root / "results/predictions.jsonl"
        records = e.read_log(path)
        records[0]["truth"] = "civilian"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            e.report()

    def test_model_loading_explicitly_disables_token_and_remote_code(self):
        from unittest.mock import Mock

        processor = types.SimpleNamespace(
            image_processor=types.SimpleNamespace(size={})
        )
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        auto_processor = types.SimpleNamespace(
            from_pretrained=Mock(return_value=processor)
        )
        auto_model = types.SimpleNamespace(
            from_pretrained=Mock(return_value=model)
        )
        module = types.SimpleNamespace(
            AutoProcessor=auto_processor,
            AutoModelForImageTextToText=auto_model,
        )
        with patch.dict(sys.modules, {"transformers": module}):
            for spec in self.config["models"]:
                self.real_load_model(spec, "cuda", "float16")
                kwargs = auto_model.from_pretrained.call_args.kwargs
                self.assertIs(kwargs["token"], False)
                self.assertIs(kwargs["trust_remote_code"], False)
                self.assertEqual(kwargs["revision"], spec["revision"])
                self.assertEqual(kwargs["attn_implementation"], "sdpa")

    def test_prediction_decodes_only_generated_tokens_and_keeps_integer_inputs(
        self,
    ):
        class Tensor:
            def __init__(self, shape, floating=False):
                self.shape = shape
                self.floating = floating
                self.dtype_used = None

            def to(self, device=None, dtype=None):
                if dtype is not None:
                    self.dtype_used = dtype
                return self

            def __getitem__(self, key):
                return Tensor((1, self.shape[1] - key[1].start))

        class Batch(dict):
            def to(self, device):
                return self

        batch = Batch(
            input_ids=Tensor((1, 10)),
            pixel_values=Tensor((1, 3), True),
            image_grid_thw=Tensor((1, 3)),
        )
        from unittest.mock import Mock

        processor = Mock()
        processor.apply_chat_template.return_value = "prompt with image token"
        processor.return_value = batch
        processor.batch_decode.side_effect = lambda tokens, **kw: (
            ["military"] if tokens.shape[1] == 2 else ["invalid prompt leaked"]
        )
        model = types.SimpleNamespace(generate=lambda **kw: Tensor((1, 12)))
        torch = types.SimpleNamespace(
            is_floating_point=lambda t: t.floating,
            inference_mode=contextlib.nullcontext,
        )
        with patch.object(e, "ROOT", self.source):
            raw, seconds, count = e._predict(
                processor,
                model,
                self.rows[0],
                self.config,
                torch,
                "cpu",
                "float32",
            )
        self.assertEqual(raw, "military")
        self.assertEqual(count, 2)
        self.assertGreaterEqual(seconds, 0)
        self.assertIsNone(batch["input_ids"].dtype_used)
        self.assertIsNone(batch["image_grid_thw"].dtype_used)
        self.assertEqual(batch["pixel_values"].dtype_used, "float32")


if __name__ == "__main__":
    unittest.main()
