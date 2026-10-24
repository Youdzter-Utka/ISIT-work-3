"""Figures for aircraft-classification results."""

from pathlib import Path


def build_figures(results_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    results_dir = Path(results_dir)
    metrics = pd.read_csv(results_dir / "metrics.csv").set_index("model")
    predictions = pd.read_json(
        results_dir / "predictions.jsonl", lines=True, dtype={"id": str}
    )
    models = [
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]
    names = ["SmolVLM-500M", "Qwen2.5-VL-3B"]
    colors = ["#b86b24", "#176b87"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.7, 1]}
        )
        x = np.arange(3)
        keys = ["accuracy", "f1_macro", "hard_accuracy"]
        for i, (model, name, color) in enumerate(zip(models, names, colors)):
            values = [metrics.loc[model, key] for key in keys]
            bars = axes[0].bar(
                x + (i - 0.5) * 0.34,
                values,
                width=0.34,
                color=color,
                label=name,
            )
            axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axes[0].set_xticks(
            x, ["Accuracy", "Macro-F1", "Accuracy\nна сложных фото"]
        )
        axes[0].set_ylim(0, 1.13)
        axes[0].set_yticks(np.linspace(0, 1, 6))
        axes[0].set_ylabel("Значение метрики")
        axes[0].set_title("Качество классификации", pad=34)
        axes[0].legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=2,
            frameon=False,
        )
        axes[0].grid(axis="y", alpha=0.18)
        axes[0].set_axisbelow(True)
        times = [
            metrics.loc[model, "mean_latency_s"] * 1000 for model in models
        ]
        bars = axes[1].bar(names, times, color=colors, width=0.55)
        axes[1].bar_label(bars, fmt="%.1f", padding=4)
        axes[1].set_ylim(0, max(times) * 1.22)
        axes[1].set_ylabel("Миллисекунды на изображение")
        axes[1].set_title("Среднее время обработки")
        axes[1].grid(axis="y", alpha=0.18)
        axes[1].set_axisbelow(True)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "Сравнение двух моделей · 30 фотографий · NVIDIA Tesla T4",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            results_dir / "comparison.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, model, name in zip(axes, models, names):
            group = predictions[predictions.model == model]
            matrix = confusion_matrix(
                group.truth,
                group.prediction,
                labels=["military", "civilian", "invalid"],
            )[:2, :]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=15)
            for y in range(2):
                for x in range(3):
                    ax.text(
                        x,
                        y,
                        str(matrix[y, x]),
                        ha="center",
                        va="center",
                        color="white" if matrix[y, x] >= 9 else "#172b4d",
                        fontsize=15,
                    )
            ax.set_xticks(range(3), ["Военный", "Гражданский", "Невалидный"])
            ax.set_yticks(range(2), ["Военный", "Гражданский"])
            ax.set_xlabel("Ответ модели")
            ax.set_ylabel("Эталон")
            ax.set_title(name, fontweight="bold")
        fig.suptitle("Матрицы ошибок", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            results_dir / "confusion_matrices_ru.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)
