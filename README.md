# Evidence-Aware and Uncertainty-Aware Safety Gating for Automated Alternative Credit Decisions

This repository supports a six-month research project on safety gating for automated alternative credit decisions. The project studies whether automated credit decision systems can be made safer when model outputs are checked against evidence quality, predictive uncertainty, and out-of-distribution signals before a final decision is issued.

## Research Problem

Alternative credit models can use nontraditional financial evidence to expand access to credit, but automated decisions can become unsafe when the input evidence is weak, incomplete, inconsistent, or outside the conditions under which the model was evaluated. A conventional risk model may still emit a confident-looking decision even when the underlying evidence is not decision-ready.

This project investigates a safety layer that can defer, escalate, or constrain automated credit decisions when the system lacks sufficient evidence or confidence.

## Research Question

Can an evidence-aware and uncertainty-aware safety gate reduce unsafe automated credit decisions compared with a conventional risk model operating without such a gate?

## Baseline System

The baseline system is a conventional automated credit decision pipeline:

1. Financial evidence is transformed into model features.
2. A risk model predicts credit risk.
3. The prediction is converted into an automated decision.

In the baseline, the risk model operates without a dedicated safety gate for evidence validity, predictive uncertainty, or out-of-distribution detection.

## Proposed System

The eventual research system will evaluate this pipeline:

1. Financial Evidence
2. Evidence Validation
3. Feature Pipeline
4. Risk Model
5. Confidence + OOD Detection
6. Safety Gate
7. Decision
8. Decision Passport
9. Borrower Explanation
10. Counterfactual / Replay
11. Audit Trail

The proposed safety gate will be studied as an intervention between model scoring and final decisioning.

## Current Scope

The current implementation adds a synthetic alternative-financial behavior layer for controlled experiments. It does not train models, expose APIs, build applications, or add deployment infrastructure.

The current scope includes:

- A clean Python 3.11 project structure.
- Minimal research dependencies.
- Reproducibility utilities for random seeds and deterministic splitting.
- A synthetic data generator for thin-file or credit-invisible applicant behavior.
- Monthly financial histories with approximately 12 months per applicant.
- Derived alternative-financial features for future model experiments.
- Validation checks for generated financial constraints.
- Basic tests confirming imports, reproducibility behavior, synthetic data constraints, and feature calculations.
- A research log for recording decisions, assumptions, and results over time.

## Synthetic Data Methodology

This repository uses synthetic data as a controlled experimental environment. It must not be interpreted as representing real borrowers or real credit outcomes.

The generator follows this structure:

1. Hidden applicant traits are sampled with correlation and overlap across populations.
2. Monthly financial behavior is generated from those hidden traits.
3. A 12-month financial history is retained for each applicant.
4. Alternative-financial features are derived from the monthly history.

The hidden traits include income stability, spending discipline, savings behavior, payment discipline, financial liquidity, and transaction activity. These latent variables are used only inside the generator and are not exposed as model features.

The monthly history includes observable alternative financial evidence such as income, expenses, savings balance, utility payments, rent payments, transaction counts, transaction amounts, balances, and cashflow. These variables are designed to be plausible for thin-file or credit-invisible applicants without relying on conventional credit history.

The processed feature table includes:

- `monthly_income_mean`
- `income_variability`
- `expense_to_income_ratio`
- `savings_consistency`
- `utility_payment_regularity`
- `rent_payment_regularity`
- `cashflow_volatility`
- `transaction_frequency`
- `average_monthly_balance`
- `minimum_monthly_balance`
- `positive_cashflow_ratio`
- `income_growth`
- `financial_buffer_ratio`

The synthetic population has three overlapping groups:

- `stable`
- `volatile`
- `borderline`

The groups intentionally overlap so future experiments are not reduced to simply identifying a synthetic label.

## Synthetic Outcome

The project uses a synthetic outcome named `financial_stress_event`. It represents a future adverse financial state in which an applicant may be unable to comfortably meet regular financial obligations. It is not a loan default label, repayment default label, or observed real-world credit outcome.

The outcome is generated from allowed alternative-financial features using a stochastic formulation:

```text
stress_score =
    intercept
    + weighted expense-to-income pressure
    + weighted income instability
    + weighted savings weakness
    + weighted payment irregularity
    + weighted cashflow volatility pressure
    + weighted financial buffer weakness
    + weighted positive-cashflow weakness
    + idiosyncratic noise

P(financial_stress_event) = sigmoid(stress_score)

financial_stress_event ~ Bernoulli(P(financial_stress_event))
```

The process does not use the synthetic population label (`stable`, `volatile`, `borderline`) and does not expose latent variables as model features. The stochastic noise is intentional so similar financial profiles can sometimes have different outcomes.

The model-ready processed dataset is:

- `data/processed/synthetic/model_dataset.csv`

It contains `applicant_id`, all allowed alternative-financial features, `financial_stress_probability`, and `financial_stress_event`.

The validation notebook `notebooks/02_outcome_validation.ipynb` checks event rate, probability distribution, feature relationships, descriptive event rates across synthetic populations, overlap, stochasticity, and whether any single feature is too predictive.

## Baseline Risk Models

The Phase 3 baseline represents a conventional automated decision pipeline:

```text
alternative financial features
    -> risk model
    -> P(financial_stress_event = 1)
    -> threshold-based automated decision
```

The baseline is a predictive benchmark, not the proposed research contribution.

The baseline uses a deterministic 70/15/15 train/validation/test split with stratification on `financial_stress_event`. It trains:

- Logistic Regression with standardized features.
- HistGradientBoostingClassifier.

The models use only the 13 allowed alternative-financial features. They do not use `applicant_id`, population labels, latent variables, `financial_stress_probability`, or conventional credit-history variables.

The initial threshold is fixed at 0.50, while validation sensitivity is reported for 0.30, 0.40, 0.50, 0.60, and 0.70. The held-out test set is evaluated with ROC-AUC, PR-AUC, accuracy, precision, recall, F1, confusion matrix, Brier score, log loss, positive prediction rate, and actual event rate.

Baseline artifacts are saved under:

- `experiments/baseline/metrics.json`
- `experiments/baseline/model_comparison.csv`
- `experiments/baseline/predictions_logistic.csv`
- `experiments/baseline/predictions_tree.csv`
- `experiments/baseline/models/`

The notebook `notebooks/03_baseline_risk_model.ipynb` runs the baseline experiment and shows calibration diagnostics without applying calibration correction.

## Probability Calibration

Phase 4 evaluates whether baseline risk probabilities are reliable. It compares raw probabilities, sigmoid / Platt calibration, and isotonic calibration for the two baseline models.

The calibration protocol is:

1. Fit baseline models on the training split.
2. Fit calibration mappings on the validation split only.
3. Evaluate raw and calibrated probabilities on the held-out test split.

The test set is not used to fit calibration parameters or choose a calibration method.

Calibration artifacts are saved under:

- `experiments/calibration/metrics.json`
- `experiments/calibration/calibration_comparison.csv`
- `experiments/calibration/calibration_predictions.csv`
- `experiments/calibration/models/`

The notebook `notebooks/04_probability_calibration.ipynb` reports Brier score, log loss, ECE, reliability diagrams, and probability distributions. ECE uses 10 fixed-width bins over `[0, 1]`; empty bins are ignored.

## Excluded Variables

Because this project studies thin-file and credit-invisible applicants, conventional credit-history variables are excluded. The synthetic data must not include credit score, CIBIL score, prior loan history, previous EMI repayment history, credit-card repayment history, number of previous loans, bureau history, or previous-loan delinquency history.

## Generate Synthetic Data

Use Python 3.11 and install the project requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the default synthetic dataset:

```bash
python -m src.data.synthetic_generator
```

This writes:

- `data/raw/synthetic/applicant_profiles.csv`
- `data/raw/synthetic/financial_history.csv`
- `data/processed/synthetic/alternative_features.csv`
- `data/processed/synthetic/metadata.json`

## Sanity Check Notebook

The notebook `notebooks/01_synthetic_data_validation.ipynb` inspects distributions, ratios, payment regularity, cashflow volatility, feature correlations, and differences across applicant populations. It is for validation only and does not train a model.

## Future Research Phases

Likely future phases include:

- Dataset selection and documentation.
- Evidence validation criteria.
- Baseline risk model implementation.
- Safety gate design and ablation studies.
- Confidence estimation and out-of-distribution detection.
- Evaluation of unsafe decision reduction.
- Decision passport and audit trail design.
- Borrower-facing explanation experiments.
- Counterfactual and replay workflows.

These phases should be added incrementally, with tests and research notes attached to each step.

## Scientific Limitations

Synthetic data can help test software behavior, experimental design, and controlled research hypotheses. It cannot by itself establish real-world validity, fairness, regulatory suitability, or borrower impact. Any future empirical claim about real borrowers will require appropriate real-world data access, documentation, validation, and governance.

## Reproducibility

The project uses a central default random seed and helper utilities in `src/utils/reproducibility.py`. Synthetic generation accepts `n_applicants`, `months`, `seed`, and `population_proportions` so runs can be repeated exactly.

## Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
