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

## Current Phase 1 Scope

Phase 1 creates the reproducible research foundation only. It does not train models, download datasets, expose APIs, build applications, or add deployment infrastructure.

Phase 1 includes:

- A clean Python 3.11 project structure.
- Minimal research dependencies.
- Reproducibility utilities for random seeds and deterministic splitting.
- Placeholder modules with docstrings for future research phases.
- Basic tests confirming imports and reproducibility behavior.
- A research log for recording decisions, assumptions, and results over time.

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

## Reproducibility

The project uses a central default random seed and helper utilities in `src/utils/reproducibility.py`. Future experiments should use these utilities for repeatable train/test splitting and randomized operations where appropriate.

## Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
