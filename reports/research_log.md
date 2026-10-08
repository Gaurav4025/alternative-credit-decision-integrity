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

## Open Research Decisions

- Select the first dataset and document licensing, provenance, variables, and known limitations.
- Define what counts as an unsafe automated credit decision for the first experiment.
- Define evidence validation rules before model training begins.
- Decide baseline model family and evaluation metrics.
