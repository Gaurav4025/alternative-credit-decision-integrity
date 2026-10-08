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

## Open Research Decisions

- Select the first dataset and document licensing, provenance, variables, and known limitations.
- Define what counts as an unsafe automated credit decision for the first experiment.
- Define evidence validation rules before model training begins.
- Decide baseline model family and evaluation metrics.
