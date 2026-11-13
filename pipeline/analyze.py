import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.storage import read
from pipeline.modeling import holdout, split

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def run():
    df = read()
    ARTIFACTS.mkdir(exist_ok=True)
    baseline, _, _ = holdout(df, "mean")
    ridge, test, predictions = holdout(df, "ridge")
    result = {"rows": len(df), "mean_baseline": baseline, "ridge": ridge,
              "splits": [{"rows": len(part), "start": str(part.date.iloc[0]), "end": str(part.date.iloc[-1])}
                         for part in split(df)]}
    (ARTIFACTS / "pipeline_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    points = axes[0].scatter(df.temperature_c, df.rentals, c=df.precipitation_mm, cmap="viridis", s=10)
    fig.colorbar(points, ax=axes[0], label="Precipitation (mm)")
    axes[0].set(xlabel="Mean daily temperature (°C)", ylabel="Daily rentals", title="Weather and demand (association)")
    axes[1].plot(test.date.astype(str), test.rentals, label="Actual", linewidth=1)
    axes[1].plot(test.date.astype(str), predictions, label="Ridge", linewidth=1)
    axes[1].set_xticks([0, len(test)//2, len(test)-1])
    axes[1].set(title="Chronological holdout", ylabel="Daily rentals")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "pipeline_analysis.png", dpi=160)
    plt.close(fig)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

