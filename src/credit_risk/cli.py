"""Run a real-data credit-default modelling experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import download_dataset, validate_and_clean
from .modeling import (
    stratified_split, fit_models, calibrate_probabilities,
    evaluate_probabilities, calibration_points, confusion_matrix_values,
)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw, checksum = download_dataset(args.raw_path, refresh=args.refresh)
    X, y = validate_and_clean(raw)
    split = stratified_split(X, y, random_state=args.seed)
    models = fit_models(split.X_train, split.y_train, random_state=args.seed)

    metrics = {}
    curves = {}
    for name, model in models.items():
        raw_prob = model.predict_proba(split.X_test)[:, 1]
        calibrated = calibrate_probabilities(model, split.X_cal, split.y_cal, method=args.calibration)
        calibrated_prob = calibrated.predict_proba(split.X_test)[:, 1]
        metrics[f"{name} | uncalibrated"] = evaluate_probabilities(split.y_test, raw_prob, args.threshold)
        metrics[f"{name} | calibrated ({args.calibration})"] = evaluate_probabilities(
            split.y_test, calibrated_prob, args.threshold
        )
        curves[f"{name} | uncalibrated"] = calibration_points(split.y_test, raw_prob)
        curves[f"{name} | calibrated"] = calibration_points(split.y_test, calibrated_prob)
        joblib.dump(model, model_dir / (name.lower().replace(" ", "_") + ".joblib"))
        joblib.dump(calibrated, model_dir / (name.lower().replace(" ", "_") + "_calibrated.joblib"))
        pd.DataFrame({
            "y_true": split.y_test,
            "probability_uncalibrated": raw_prob,
            "probability_calibrated": calibrated_prob,
        }, index=split.y_test.index).to_csv(
            output / (name.lower().replace(" ", "_") + "_test_probabilities.csv"), index_label="row_id"
        )
        cm = confusion_matrix_values(split.y_test, calibrated_prob, args.threshold)
        pd.DataFrame(cm, index=["actual_nondefault", "actual_default"],
                     columns=["predicted_nondefault", "predicted_default"]).to_csv(
                         output / (name.lower().replace(" ", "_") + "_confusion_matrix.csv")
                     )

    metric_df = pd.DataFrame(metrics).T
    metric_df.index.name = "model"
    metric_df.to_csv(output / "metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (predicted, observed) in curves.items():
        axes[0].plot(predicted, observed, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted default probability", ylabel="Observed default rate",
                title="Reliability diagram")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    metric_df[["ROC_AUC", "PR_AUC_average_precision"]].plot(kind="bar", ax=axes[1])
    axes[1].set(title="Ranking performance", ylabel="Score", ylim=(0, 1))
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)

    report = {
        "source": "UCI Machine Learning Repository — Default of Credit Card Clients",
        "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "sha256_raw_dataset": checksum,
        "rows": int(len(raw)),
        "features": int(X.shape[1]),
        "default_rate": float(y.mean()),
        "train_rows": int(len(split.y_train)),
        "calibration_rows": int(len(split.y_cal)),
        "test_rows": int(len(split.y_test)),
        "random_seed": args.seed,
        "calibration_method": args.calibration,
        "threshold": args.threshold,
        "notes": [
            "Random stratified split, not a temporal split: use only for a baseline project.",
            "Dataset is historical and from Taiwan; it is not a current lending population.",
            "Threshold 0.5 is a reporting convention, not an optimal lending decision threshold.",
        ],
    }
    (output / "run_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Dataset: UCI Default of Credit Card Clients (real historical records)")
    print(f"Rows={len(raw):,}; features={X.shape[1]}; default rate={y.mean():.3%}")
    print(f"Split sizes: train={len(split.y_train):,}, calibration={len(split.y_cal):,}, test={len(split.y_test):,}")
    print("\nMetrics:")
    print(metric_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to {output.resolve()} and models to {model_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default="data/raw/default_of_credit_card_clients.xls")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--refresh", action="store_true", help="Download original data again")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must lie strictly between 0 and 1")
    main_run = run
    main_run(args)


if __name__ == "__main__":
    main()
