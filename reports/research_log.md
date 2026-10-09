# Research Log

## 2026-10-08

- Initialized the reproducible research foundation for the project.
- Confirmed that Phase 1 is limited to structure, documentation, reproducibility utilities, and import tests.
- Deferred dataset download, model training, application development, APIs, RAG, LLMs, agents, containerization, orchestration, and deployment.
- Added the synthetic alternative-financial behavior layer for thin-file and credit-invisible applicant experiments.
- Implemented a generator that samples hidden applicant traits, creates monthly financial histories, and derives alternative-credit features.
- Preserved monthly history separately from applicant-level features to support later temporal analysis and evidence-quality experiments.

## Synthetic Data Rationale

Synthetic data is being used to create a controlled experimental environment before real data access, governance, and validation are available. It lets the project test reproducibility, feature definitions, temporal structure, evidence-quality assumptions, and future safety-gating experimental design without claiming that the records represent real borrowers.

Conventional credit-history variables are excluded because the research focuses on thin-file and credit-invisible applicants. The synthetic data does not include credit score, CIBIL score, previous loan history, previous EMI repayment history, credit-card repayment history, number of previous loans, bureau history, or previous-loan delinquency history.

Alternative financial evidence in this research currently means observable financial behavior that could plausibly exist without a conventional credit file: income deposits, expense patterns, savings balances, utility bill amounts and payment status, rent amounts and payment status, transaction counts, transaction amounts, account balances, and cashflow.

The generator uses latent variables for income stability, spending discipline, savings behavior, payment discipline, financial liquidity, and transaction activity. These latent variables create correlated observable behavior, but they are not exported as model features.

Temporal financial histories are generated month by month. Income varies according to income stability, expenses are related to income and spending discipline, rent is relatively stable, utility bills vary modestly, savings evolve from net cashflow, and transactions vary by applicant activity.

Derived features are calculated from the monthly history rather than independently sampled. Current features summarize income level and variability, expense-to-income behavior, savings consistency, payment regularity, cashflow volatility, transaction frequency, balances, positive cashflow share, income growth, and financial buffer ratio.

The main limitation is that synthetic data reflects generator assumptions. It can support controlled experiments and software validation, but it cannot by itself establish real-world validity, borrower impact, fairness, compliance, or deployment readiness.

Synthetic data also creates the risk of accidentally making future safety-gate results look better than they are. To reduce this risk, the current groups intentionally overlap and the generator does not create a target label or hard-code future model outcomes.

## Synthetic Financial Stress Outcome

The research needs an outcome variable so later experiments can compare conventional automated decisions against gated decisions. Because the synthetic applicants are designed as thin-file or credit-invisible applicants, it would be scientifically misleading to create a target named loan default, repayment default, or observed credit default. The synthetic population has no real loan histories and no observed repayment outcomes.

The current outcome is named `financial_stress_event`. It is defined as a synthetic future adverse financial state indicating severe financial stress or inability to comfortably meet regular financial obligations. It is not equivalent to real-world credit default and should not be described as such.

The outcome is generated from observable alternative-financial features:

```text
stress_score =
    -1.85
    + 1.10 * expense_pressure
    + 0.90 * income_instability
    + 0.85 * savings_weakness
    + 1.00 * payment_irregularity
    + 0.80 * cashflow_volatility_pressure
    + 1.10 * buffer_weakness
    + 0.80 * cashflow_weakness
    + Normal(0, 0.65)

financial_stress_probability = sigmoid(stress_score)

financial_stress_event ~ Bernoulli(financial_stress_probability)
```

The synthetic population label is not used in this calculation. Latent applicant traits are not exposed in the model dataset. Any influence from underlying applicant behavior enters through derived financial features such as expense-to-income pressure, income variability, savings consistency, payment regularity, cashflow volatility, and financial buffer ratio.

The formulation is intentionally stochastic. Similar applicants can receive different outcomes, which reflects uncertainty in future financial conditions and helps avoid deterministic threshold labels. It is also intentionally not optimized to make a future model or safety gate look successful.

Limitations remain substantial. The event is a generated research construct based on assumptions, not observed borrower behavior. It can support controlled experiments about software behavior and methodology, but it cannot establish real-world credit risk validity, fairness, compliance, or borrower impact.

## Baseline Risk Model

The baseline is required so later safety-gating experiments have a conventional automated decision benchmark. It represents the ordinary path of alternative financial features to a risk probability and automated decision. The baseline is a predictive benchmark, not the proposed research contribution.

The baseline uses `data/processed/synthetic/model_dataset.csv` and predicts `financial_stress_event`. It explicitly excludes `applicant_id`, synthetic population labels, latent variables, `financial_stress_probability`, and conventional credit-history variables.

The feature set contains the 13 allowed alternative-financial features:

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

The split strategy is deterministic 70/15/15 train/validation/test with stratification on `financial_stress_event`. The validation set is used only for threshold sensitivity analysis. The held-out test set is used for final metrics.

Two scikit-learn models were trained:

- Logistic Regression with standardized features.
- HistGradientBoostingClassifier as a lightweight tree-based baseline.

Both models output `P(financial_stress_event = 1)`. The initial decision threshold is documented as 0.50, with validation-set sensitivity analysis at 0.30, 0.40, 0.50, 0.60, and 0.70. Lower thresholds increase recall and positive prediction rate; higher thresholds are more selective but miss more stress events.

Held-out test results:

| Model | ROC-AUC | PR-AUC | Brier | Log Loss | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6582 | 0.5397 | 0.2081 | 0.6055 | 0.6347 | 0.2658 | 0.3747 |
| HistGradientBoostingClassifier | 0.6512 | 0.5254 | 0.2107 | 0.6130 | 0.5738 | 0.2677 | 0.3651 |

The test actual event rate is approximately 0.3487. At the 0.50 threshold, Logistic Regression has a positive prediction rate of 0.1460 and HistGradientBoosting has a positive prediction rate of 0.1627.

Calibration diagnostics were produced through reliability-diagram bins and Brier score. No calibration correction was applied. The Brier scores around 0.21 suggest probability quality is imperfect and should be studied before any safety-gating claims are made.

Leakage review: performance is moderate rather than unexpectedly high. The feature list excludes the generated stress probability and identifiers. However, the outcome is synthetic and generated from the same family of financial features, so performance cannot be interpreted as real-world credit-risk validity.

## Probability Calibration

Calibration is relevant because downstream automated decision-making and future safety-gating research depend on probability reliability, not only ranking performance. A model can have useful ROC-AUC while assigning probabilities that do not match observed event rates.

The calibration phase evaluates the two existing baseline models without redesigning them:

- Logistic Regression.
- HistGradientBoostingClassifier.

Three probability variants are evaluated for each model:

- Raw uncalibrated probabilities.
- Sigmoid / Platt calibration.
- Isotonic calibration.

Protocol: baseline models are fit on the training set. Sigmoid and isotonic calibration mappings are fit on the validation set only. The held-out test set is used only for final reporting. Test labels are not used to fit calibration parameters or choose the calibration method.

Expected Calibration Error uses 10 fixed-width probability bins over `[0, 1]`. For each non-empty bin, the contribution is the bin sample share multiplied by the absolute gap between mean predicted probability and observed event rate. Empty bins are ignored.

Held-out test probability-quality results:

| Model | Method | Brier | Log Loss | ECE |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | Raw | 0.2081 | 0.6055 | 0.0210 |
| Logistic Regression | Sigmoid | 0.2085 | 0.6066 | 0.0244 |
| Logistic Regression | Isotonic | 0.2094 | 0.6517 | 0.0155 |
| HistGradientBoostingClassifier | Raw | 0.2107 | 0.6130 | 0.0379 |
| HistGradientBoostingClassifier | Sigmoid | 0.2108 | 0.6116 | 0.0258 |
| HistGradientBoostingClassifier | Isotonic | 0.2120 | 0.6592 | 0.0217 |

Calibration observations: logistic regression was already fairly calibrated by ECE, and sigmoid calibration slightly worsened Brier, log loss, and ECE. Isotonic reduced ECE but worsened log loss and ranking metrics. For HistGradientBoosting, both sigmoid and isotonic improved ECE, while Brier score did not improve and isotonic substantially worsened log loss. This is a useful mixed result rather than evidence that calibration universally improves all probability metrics.

Implications for the future Safety Gate: calibrated probability estimates may help downstream decision policies, but calibration alone is not uncertainty estimation, OOD detection, or evidence-quality scoring. Safety-gate experiments should treat calibration as one input to reliability analysis, not as a replacement for uncertainty-aware gating.

Limitations: all results remain synthetic and depend on the generated outcome mechanism. Calibration quality must be reassessed under distribution shift, evidence corruption, and eventually real-world validation data before drawing practical credit-decision conclusions.

## Open Research Decisions

- Select the first dataset and document licensing, provenance, variables, and known limitations.
- Define what counts as an unsafe automated credit decision for the first experiment.
- Define evidence validation rules before model training begins.
- Decide baseline model family and evaluation metrics.
