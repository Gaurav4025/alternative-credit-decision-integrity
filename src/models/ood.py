"""Training-only OOD detectors for the permitted alternative-financial features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data.ood_scenarios import OOD_SCENARIOS, OOD_SEVERITIES, OODScenarioConfig, generate_ood_feature_cohort
from src.evaluation.ood_metrics import evaluate_ood_detection, select_ood_threshold
from src.features.alternative_features import FEATURE_COLUMNS
from src.models.baseline import load_model_dataset, make_train_validation_test_split
from src.utils.reproducibility import DEFAULT_RANDOM_SEED

OOD_FEATURES = tuple(FEATURE_COLUMNS)
OOD_THRESHOLD_PERCENTILE = 0.95
PROHIBITED_OOD_INPUT_COLUMNS = {
    "applicant_id",
    "financial_stress_event",
    "financial_stress_probability",
    "population_group",
    "predicted_probability",
}


class OODDetector(Protocol):
    """Common detector interface using higher scores for more OOD-like inputs."""

    def fit(self, features: pd.DataFrame) -> "OODDetector": ...

    def score_samples(self, features: pd.DataFrame) -> np.ndarray: ...


def validate_ood_features(features: pd.DataFrame) -> pd.DataFrame:
    """Require exactly the permitted 13-feature OOD input contract."""

    prohibited = PROHIBITED_OOD_INPUT_COLUMNS.intersection(features.columns)
    if prohibited:
        raise ValueError(f"Prohibited columns in OOD detector input: {sorted(prohibited)}")
    missing = set(OOD_FEATURES) - set(features.columns)
    unexpected = set(features.columns) - set(OOD_FEATURES)
    if missing or unexpected:
        raise ValueError(
            "OOD detector inputs must contain exactly the permitted features; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    result = features.loc[:, OOD_FEATURES].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("OOD detector inputs must be finite numeric values.")
    return result


@dataclass
class StandardizedEuclideanDetector:
    """Distance from the training center after training-only standardization."""

    scaler: StandardScaler | None = None
    n_reference_samples: int = 0

    def fit(self, features: pd.DataFrame) -> "StandardizedEuclideanDetector":
        matrix = validate_ood_features(features)
        self.scaler = StandardScaler().fit(matrix)
        self.n_reference_samples = len(matrix)
        return self

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Detector must be fit before scoring.")
        return np.linalg.norm(self.scaler.transform(validate_ood_features(features)), axis=1)


@dataclass
class ShrinkageMahalanobisDetector:
    """Ledoit-Wolf Mahalanobis distance with a train-only standardized reference."""

    scaler: StandardScaler | None = None
    covariance: LedoitWolf | None = None
    n_reference_samples: int = 0

    def fit(self, features: pd.DataFrame) -> "ShrinkageMahalanobisDetector":
        matrix = validate_ood_features(features)
        self.scaler = StandardScaler().fit(matrix)
        self.covariance = LedoitWolf().fit(self.scaler.transform(matrix))
        self.n_reference_samples = len(matrix)
        return self

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.covariance is None:
            raise RuntimeError("Detector must be fit before scoring.")
        transformed = self.scaler.transform(validate_ood_features(features))
        centered = transformed - self.covariance.location_
        squared = np.einsum("ij,jk,ik->i", centered, self.covariance.precision_, centered)
        return np.sqrt(np.maximum(squared, 0.0))


@dataclass
class IsolationForestDetector:
    """Fixed-seed Isolation Forest with scores oriented toward OOD likelihood."""

    seed: int = DEFAULT_RANDOM_SEED
    scaler: StandardScaler | None = None
    forest: IsolationForest | None = None
    n_reference_samples: int = 0

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        matrix = validate_ood_features(features)
        self.scaler = StandardScaler().fit(matrix)
        self.forest = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=self.seed,
        ).fit(self.scaler.transform(matrix))
        self.n_reference_samples = len(matrix)
        return self

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.forest is None:
            raise RuntimeError("Detector must be fit before scoring.")
        return -self.forest.score_samples(self.scaler.transform(validate_ood_features(features)))


@dataclass(frozen=True)
class OODExperimentConfig:
    """Configuration for deterministic OOD experiments."""

    seed: int = DEFAULT_RANDOM_SEED
    threshold_percentile: float = OOD_THRESHOLD_PERCENTILE


def fit_ood_detectors(train_features: pd.DataFrame, seed: int = DEFAULT_RANDOM_SEED) -> dict[str, OODDetector]:
    """Fit every detector exclusively on clean training features."""

    return {
        "standardized_euclidean": StandardizedEuclideanDetector().fit(train_features),
        "shrinkage_mahalanobis": ShrinkageMahalanobisDetector().fit(train_features),
        "isolation_forest": IsolationForestDetector(seed=seed).fit(train_features),
    }


def run_ood_experiment(
    dataset_path: str | Path = "data/processed/synthetic/model_dataset.csv",
    history_path: str | Path = "data/raw/synthetic/financial_history.csv",
    output_dir: str | Path = "experiments/ood",
    config: OODExperimentConfig | None = None,
) -> dict[str, object]:
    """Run train-only OOD fitting, validation thresholds, and test evaluation."""

    config = config or OODExperimentConfig()
    dataset = load_model_dataset(dataset_path)
    splits = make_train_validation_test_split(dataset, seed=config.seed)
    _validate_splits(splits, dataset)
    history = pd.read_csv(history_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _persist_split_ids(splits, output_path)

    train_features = splits["train"].loc[:, OOD_FEATURES]
    detectors = fit_ood_detectors(train_features, seed=config.seed)
    validation_features = splits["validation"].loc[:, OOD_FEATURES]
    test_features = splits["test"].loc[:, OOD_FEATURES]
    thresholds = {
        name: select_ood_threshold(detector.score_samples(validation_features), config.threshold_percentile)
        for name, detector in detectors.items()
    }

    score_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    scenario_config = OODScenarioConfig(seed=config.seed)
    for split_name, split, clean_features in (
        ("validation", splits["validation"], validation_features),
        ("test", splits["test"], test_features),
    ):
        clean_ids = split["applicant_id"].to_numpy()
        clean_scores = _score_clean(detectors, clean_features, clean_ids, split_name, thresholds)
        score_rows.append(clean_scores)
        for scenario_index, scenario in enumerate(OOD_SCENARIOS):
            for severity_index, severity in enumerate(OOD_SEVERITIES):
                scenario_seed = np.random.SeedSequence(config.seed).spawn(
                    len(OOD_SCENARIOS) * len(OOD_SEVERITIES)
                )[scenario_index * len(OOD_SEVERITIES) + severity_index]
                cohort = generate_ood_feature_cohort(
                    history,
                    clean_ids,
                    scenario,
                    severity,
                    seed=int(scenario_seed.generate_state(1)[0]),
                )
                cohort_features = cohort.loc[:, OOD_FEATURES]
                scenario_scores = _score_ood(
                    detectors, cohort_features, cohort, split_name, thresholds
                )
                score_rows.append(scenario_scores)
                for detector_name in detectors:
                    clean_values = clean_scores.loc[
                        clean_scores["detector"] == detector_name, "ood_score"
                    ]
                    ood_values = scenario_scores.loc[
                        scenario_scores["detector"] == detector_name, "ood_score"
                    ]
                    metrics = evaluate_ood_detection(
                        clean_values,
                        ood_values,
                        thresholds[detector_name],
                    )
                    summary_rows.append(
                        {
                            "evaluation_split": split_name,
                            "detector": detector_name,
                            "scenario": scenario,
                            "severity": severity,
                            "threshold": thresholds[detector_name],
                            **metrics,
                        }
                    )

    scores = pd.concat(score_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    scores.to_csv(output_path / "ood_scores.csv", index=False)
    summary.to_csv(output_path / "scenario_summary.csv", index=False)

    metadata = {
        "config": asdict(config),
        "feature_contract": list(OOD_FEATURES),
        "reference_population": "clean training split only",
        "split_strategy": "existing deterministic 70/15/15 stratified split with seed 42",
        "threshold_protocol": "95th percentile of clean validation scores only",
        "thresholds": thresholds,
        "score_convention": "higher scores are more OOD-like",
        "scenario_config": scenario_config.as_dict(),
        "forbidden_inputs": sorted(PROHIBITED_OOD_INPUT_COLUMNS),
        "uses_outcomes_or_probabilities": False,
        "artificial_ood_prevalence": 0.5,
        "limitations": (
            "Controlled synthetic shifts test detector mechanics only; they do not establish "
            "real-world OOD detection validity or financial-stress prediction performance."
        ),
    }
    (output_path / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _score_clean(
    detectors: dict[str, OODDetector],
    features: pd.DataFrame,
    applicant_ids: np.ndarray,
    split_name: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    frames = []
    for name, detector in detectors.items():
        scores = detector.score_samples(features)
        frames.append(pd.DataFrame({
            "evaluation_split": split_name,
            "detector": name,
            "source_applicant_id": applicant_ids,
            "scenario": "clean_reference",
            "severity": "none",
            "is_controlled_ood": False,
            "evidence_quality_status": "clean_verified",
            "ood_score": scores,
            "threshold": thresholds[name],
            "is_ood_flag": scores >= thresholds[name],
        }))
    return pd.concat(frames, ignore_index=True)


def _score_ood(
    detectors: dict[str, OODDetector],
    features: pd.DataFrame,
    cohort: pd.DataFrame,
    split_name: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    frames = []
    for name, detector in detectors.items():
        scores = detector.score_samples(features)
        frames.append(pd.DataFrame({
            "evaluation_split": split_name,
            "detector": name,
            "source_applicant_id": cohort["source_applicant_id"].to_numpy(),
            "scenario": cohort["ood_scenario"].to_numpy(),
            "severity": cohort["ood_severity"].to_numpy(),
            "is_controlled_ood": True,
            "evidence_quality_status": cohort["evidence_quality_status"].to_numpy(),
            "ood_score": scores,
            "threshold": thresholds[name],
            "is_ood_flag": scores >= thresholds[name],
        }))
    return pd.concat(frames, ignore_index=True)


def _persist_split_ids(splits: dict[str, pd.DataFrame], output_path: Path) -> None:
    split_ids = pd.concat(
        [frame[["applicant_id"]].assign(split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    split_ids.to_csv(output_path / "split_applicant_ids.csv", index=False)


def _validate_splits(splits: dict[str, pd.DataFrame], dataset: pd.DataFrame) -> None:
    id_sets = {name: set(frame["applicant_id"]) for name, frame in splits.items()}
    if id_sets["train"] & id_sets["validation"] or id_sets["train"] & id_sets["test"] or id_sets["validation"] & id_sets["test"]:
        raise ValueError("Train, validation, and test applicant IDs must be disjoint.")
    if set.union(*id_sets.values()) != set(dataset["applicant_id"]):
        raise ValueError("Every applicant must appear in exactly one split.")


if __name__ == "__main__":
    run_ood_experiment()
