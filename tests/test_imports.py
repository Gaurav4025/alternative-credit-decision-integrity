def test_project_modules_import() -> None:
    import src.data.validation
    import src.evaluation.metrics
    import src.features.pipeline
    import src.models.risk_model
    import src.utils.config
    import src.utils.reproducibility

    assert src.utils.reproducibility.DEFAULT_RANDOM_SEED == 42
