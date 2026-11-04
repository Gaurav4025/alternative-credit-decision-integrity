"""Stratified-bootstrap prediction uncertainty for existing baseline models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.uncertainty_metrics import (
    ensemble_stability,
    evaluate_uncertainty,
    select_uncertainty_threshold,
    summarize_member_predictions,
)
from src.features.alternative_features import FEATURE_COLUMNS
from src.models.baseline import TARGET_COLUMN, load_model_dataset
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

UNCERTAINTY_FEATURES = tuple(FEATURE_COLUMNS)
MODEL_FAMILIES = ("logistic_regression", "hist_gradient_boosting")
INITIAL_ENSEMBLE_SIZE = 50
EXPANDED_ENSEMBLE_SIZE = 100
PROHIBITED_UNCERTAINTY_INPUT_COLUMNS = {
    "applicant_id",
    "financial_stress_event",
    "financial_stress_probability",
    "population_group",
    "predicted_probability",
    "ood_score",
}


@dataclass(frozen=True)
class UncertaintyConfig:
    """Deterministic settings for bootstrap uncertainty experiments."""

    seed: int = DEFAULT_RANDOM_SEED
    initial_ensemble_size: int = INITIAL_ENSEMBLE_SIZE
    expanded_ensemble_size: int = EXPANDED_ENSEMBLE_SIZE
    threshold_percentile: float = 0.95


@dataclass
class BootstrapEnsemble:
    """Fitted members and provenance for one model-family bootstrap ensemble."""

    model_family: str
    members: list[Any]
    member_seeds: list[int]
    class_sample_counts: list[dict[int, int]]
    training_applicant_ids: set[int]


def validate_uncertainty_features(features: pd.DataFrame) -> pd.DataFrame:
    """Require exactly the existing 13 permitted uncertainty-model features."""

    prohibited = PROHIBITED_UNCERTAINTY_INPUT_COLUMNS.intersection(features.columns)
    if prohibited:
        raise ValueError(f"Prohibited columns in uncertainty-model input: {sorted(prohibited)}")
    missing = set(UNCERTAINTY_FEATURES) - set(features.columns)
    unexpected = set(features.columns) - set(UNCERTAINTY_FEATURES)
    if missing or unexpected:
        raise ValueError(
            "Uncertainty-model inputs must contain exactly the permitted features; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    result = features.loc[:, UNCERTAINTY_FEATURES].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Uncertainty-model inputs must be finite numeric values.")
    return result


def stratified_bootstrap_sample(train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[int, int]]:
    """Resample each training class with replacement, preserving class counts."""

    rng = np.random.default_rng(seed)
    samples = []
    class_counts: dict[int, int] = {}
    for label, group in train.groupby(TARGET_COLUMN, sort=True):
        sampled_indexes = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        samples.append(train.loc[sampled_indexes])
        class_counts[int(label)] = len(group)
    sampled = pd.concat(samples, ignore_index=True)
    order = rng.permutation(len(sampled))
    return sampled.iloc[order].reset_index(drop=True), class_counts


def fit_bootstrap_ensemble(
    train: pd.DataFrame,
    model_family: str,
    member_seeds: list[int],
) -> BootstrapEnsemble:
    """Fit deterministic bootstrap members using only the supplied training split."""

    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {model_family}")
    _validate_training_dataset(train)
    members: list[Any] = []
    class_sample_counts: list[dict[int, int]] = []
    for seed in member_seeds:
        sample, class_counts = stratified_bootstrap_sample(train, seed)
        model = _make_model(model_family, seed)
        model.fit(sample.loc[:, UNCERTAINTY_FEATURES], sample[TARGET_COLUMN])
        members.append(model)
        class_sample_counts.append(class_counts)
    return BootstrapEnsemble(
        model_family=model_family,
        members=members,
        member_seeds=member_seeds,
        class_sample_counts=class_sample_counts,
        training_applicant_ids=set(train["applicant_id"]),
    )


def predict_bootstrap_ensemble(ensemble: BootstrapEnsemble, features: pd.DataFrame) -> np.ndarray:
    """Return an applicant-by-member matrix of raw bootstrap probabilities."""

    matrix = validate_uncertainty_features(features)
    probabilities = np.column_stack([member.predict_proba(matrix)[:, 1] for member in ensemble.members])
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Bootstrap members returned invalid probabilities.")
    return probabilities


def run_uncertainty_experiment(
    dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    split_ids_path: str | Path = "experiments/ood/split_applicant_ids.csv",
    output_dir: str | Path = "experiments/uncertainty",
    config: UncertaintyConfig | None = None,
) -> dict[str, object]:
    """Fit train-only ensembles, select validation thresholds, and evaluate once."""

    config = config or UncertaintyConfig()
    dataset = load_model_dataset(dataset_path)
    splits = load_persisted_splits(dataset, split_ids_path)
    seed_sequence = np.random.SeedSequence(config.seed)
    child_sequences = seed_sequence.spawn(config.expanded_ensemble_size * len(MODEL_FAMILIES))
    family_seeds = {
        family: [
            int(child.generate_state(1)[0])
            for child in child_sequences[
                index * config.expanded_ensemble_size : (index + 1) * config.expanded_ensemble_size
            ]
        ]
        for index, family in enumerate(MODEL_FAMILIES)
    }
    ensembles = {
        family: fit_bootstrap_ensemble(
            splits["train"], family, family_seeds[family][:config.initial_ensemble_size]
        )
        for family in MODEL_FAMILIES
    }

    validation_probabilities = {
        family: predict_bootstrap_ensemble(ensemble, splits["validation"].loc[:, UNCERTAINTY_FEATURES])
        for family, ensemble in ensembles.items()
    }
    stability = {
        family: ensemble_stability(probabilities[:, : config.initial_ensemble_size // 2], probabilities)
        for family, probabilities in validation_probabilities.items()
    }
    chosen_size = config.initial_ensemble_size
    if not all(result["stable"] for result in stability.values()):
        chosen_size = config.expanded_ensemble_size
        for family, ensemble in ensembles.items():
            expanded = fit_bootstrap_ensemble(
                splits["train"], family, family_seeds[family][config.initial_ensemble_size:]
            )
            ensemble.members.extend(expanded.members)
            ensemble.member_seeds.extend(expanded.member_seeds)
            ensemble.class_sample_counts.extend(expanded.class_sample_counts)
        validation_probabilities = {
            family: predict_bootstrap_ensemble(ensemble, splits["validation"].loc[:, UNCERTAINTY_FEATURES])
            for family, ensemble in ensembles.items()
        }
        stability = {
            family: ensemble_stability(probabilities[:, : config.initial_ensemble_size], probabilities)
            for family, probabilities in validation_probabilities.items()
        }

    validation_summaries = {
        family: summarize_member_predictions(probabilities)
        for family, probabilities in validation_probabilities.items()
    }
    thresholds = {
        family: select_uncertainty_threshold(summary["predictive_variance"], config.threshold_percentile)
        for family, summary in validation_summaries.items()
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    for split_name in ("validation", "test"):
        split = splits[split_name]
        member_probabilities = {
            family: validation_probabilities[family]
            if split_name == "validation"
            else predict_bootstrap_ensemble(ensemble, split.loc[:, UNCERTAINTY_FEATURES])
            for family, ensemble in ensembles.items()
        }
        family_summaries = {
            family: summarize_member_predictions(probabilities)
            for family, probabilities in member_probabilities.items()
        }
        disagreement = np.abs(
            family_summaries["logistic_regression"]["mean_probability"]
            - family_summaries["hist_gradient_boosting"]["mean_probability"]
        )
        for family in MODEL_FAMILIES:
            statistics = family_summaries[family]
            score_frames.append(pd.DataFrame({
                "evaluation_split": split_name,
                "model_family": family,
                "applicant_id": split["applicant_id"].to_numpy(),
                "mean_probability": statistics["mean_probability"],
                "predictive_variance": statistics["predictive_variance"],
                "predictive_std": statistics["predictive_std"],
                "predictive_entropy": statistics["predictive_entropy"],
                "interval_lower_05": statistics["interval_lower_05"],
                "interval_upper_95": statistics["interval_upper_95"],
                "model_family_disagreement": disagreement,
                "uncertainty_threshold": thresholds[family],
                "is_high_uncertainty": statistics["predictive_variance"] >= thresholds[family],
            }))
            member_frames.append(_member_prediction_frame(
                split_name, family, split["applicant_id"].to_numpy(), member_probabilities[family], ensembles[family].member_seeds
            ))
            metrics, curve = evaluate_uncertainty(
                split[TARGET_COLUMN].to_numpy(),
                statistics["mean_probability"],
                statistics["predictive_variance"],
                thresholds[family],
            )
            summary_rows.append({
                "evaluation_split": split_name,
                "model_family": family,
                "ensemble_size": chosen_size,
                "threshold": thresholds[family],
                **metrics,
            })
            curve_frames.append(curve.assign(evaluation_split=split_name, model_family=family))

    pd.concat(member_frames, ignore_index=True).to_csv(output_path / "ensemble_predictions.csv", index=False)
    pd.concat(score_frames, ignore_index=True).to_csv(output_path / "uncertainty_scores.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_path / "test_summary.csv", index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(output_path / "selective_prediction_curve.csv", index=False)
    threshold_metadata = {
        "threshold_percentile": config.threshold_percentile,
        "thresholds": thresholds,
        "validation_flag_rates": {
            family: float((summary["predictive_variance"] >= thresholds[family]).mean())
            for family, summary in validation_summaries.items()
        },
        "ensemble_size": chosen_size,
        "member_seeds": {
            family: family_seeds[family][:chosen_size]
            for family in MODEL_FAMILIES
        },
    }
    (output_path / "validation_thresholds.json").write_text(
        json.dumps(threshold_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "config": asdict(config),
        "feature_contract": list(UNCERTAINTY_FEATURES),
        "split_source": str(split_ids_path),
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "ensemble_size": chosen_size,
        "stability": stability,
        "threshold_protocol": "95th percentile of clean validation predictive variance only",
        "thresholds": thresholds,
        "uses_raw_probabilities": True,
        "uses_phase_four_calibrators": False,
        "uses_ood_scores": False,
        "limitations": (
            "Bootstrap variance is a training-resampling sensitivity proxy, not calibrated confidence or "
            "a complete decomposition of epistemic and aleatoric uncertainty."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_persisted_splits(dataset: pd.DataFrame, split_ids_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load and validate the exact Phase 6 persisted applicant-ID split."""

    split_ids = pd.read_csv(split_ids_path)
    if set(split_ids.columns) != {"applicant_id", "split"}:
        raise ValueError("Persisted split IDs must contain exactly applicant_id and split columns.")
    expected_sizes = {"train": 7000, "validation": 1500, "test": 1500}
    if split_ids["applicant_id"].duplicated().any() or split_ids["applicant_id"].isna().any():
        raise ValueError("Persisted split applicant IDs must be unique and non-null.")
    if split_ids["split"].value_counts().to_dict() != expected_sizes:
        raise ValueError("Persisted split sizes must match the original 7000/1500/1500 split.")
    if set(split_ids["applicant_id"]) != set(dataset["applicant_id"]):
        raise ValueError("Persisted split IDs must cover every model-dataset applicant exactly once.")
    return {
        name: dataset.merge(
            split_ids.loc[split_ids["split"] == name, ["applicant_id"]], on="applicant_id", how="inner"
        ).sort_values("applicant_id").reset_index(drop=True)
        for name in expected_sizes
    }


def _validate_training_dataset(train: pd.DataFrame) -> None:
    required = {"applicant_id", TARGET_COLUMN, *UNCERTAINTY_FEATURES}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Missing bootstrap training columns: {sorted(missing)}")
    validate_uncertainty_features(train.loc[:, UNCERTAINTY_FEATURES])
    if not set(train[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Bootstrap training targets must be binary.")


def _make_model(model_family: str, seed: int) -> Any:
    if model_family == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1_000, random_state=seed, solver="lbfgs")),
        ])
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=seed,
    )


def _member_prediction_frame(
    split_name: str,
    family: str,
    applicant_ids: np.ndarray,
    probabilities: np.ndarray,
    seeds: list[int],
) -> pd.DataFrame:
    frame = pd.DataFrame(probabilities, columns=range(len(seeds)))
    frame.insert(0, "applicant_id", applicant_ids)
    long = frame.melt(id_vars="applicant_id", var_name="member_index", value_name="member_probability")
    long["member_seed"] = long["member_index"].map(dict(enumerate(seeds)))
    long.insert(0, "model_family", family)
    long.insert(0, "evaluation_split", split_name)
    return long


if __name__ == "__main__":
    run_uncertainty_experiment()
