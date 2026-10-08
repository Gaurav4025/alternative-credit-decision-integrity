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
