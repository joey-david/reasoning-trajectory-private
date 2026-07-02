"""Load orchestration job adapters by module name."""

from __future__ import annotations

from importlib import import_module

from src.orchestration.jobs.contract import OrchestrationJob


def load_job(name: str) -> OrchestrationJob:
    """Load ``src.orchestration.jobs.<name>`` after validating its contract.

    Args:
        name: Registered orchestration job name.

    Returns:
        The imported orchestration job module.
    """
    module_name = name.replace("-", "_")
    if not module_name.isidentifier() or module_name.startswith("_"):
        raise ValueError(f"Invalid orchestration job name: {name!r}")
    module = import_module(f"src.orchestration.jobs.{module_name}")
    missing = [
        attribute
        for attribute in ("pending_tasks", "setup_worker", "log_path")
        if not callable(getattr(module, attribute, None))
    ]
    if missing:
        raise TypeError(f"Job {name!r} is missing: {', '.join(missing)}")
    return module
