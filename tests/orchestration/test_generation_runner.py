from pathlib import Path
from unittest.mock import patch

from src.orchestration.generation_runner import generate_one_run


def test_generate_one_run_limits_loaded_samples() -> None:
    config = {"dataset": {}, "model": {"device_map": {"": 0}}}
    samples = [{"id": index} for index in range(3)]

    with (
        patch("src.orchestration.generation_runner.load_config", return_value=config),
        patch(
            "src.orchestration.generation_runner.load_run_samples",
            return_value=samples,
        ),
        patch("src.models.generation_pipeline.generate_run") as generate_run,
    ):
        generate_one_run(Path("run"), limit=1)

    generate_run.assert_called_once()
    assert generate_run.call_args.args[2] == samples[:1]
