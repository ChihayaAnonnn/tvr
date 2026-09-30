"""CPU lifecycle fault fixtures; these do not satisfy four-GPU Stage 0 qualification."""

import os
import signal

import numpy as np

from experiment_artifacts import verify_array_reference

from .artifacts import fail_run, publish_metrics


def run_cpu_qualification(context, *, scenario):
    if (
        scenario not in {"success", "array_tamper", "signal"}
        or context.manifest["config"]["evaluation"].get("qualification_scenario") != scenario
    ):
        raise ValueError("qualification scenario differs from reservation")
    if (
        context.manifest["stage"] != "baseline"
        or context.manifest["test_authorized"]
        or context.manifest["input_checkpoints"]
        or context.manifest["dataset_splits"]
    ):
        raise ValueError("CPU qualification requires a validation-only baseline without model/data inputs")
    if context.manifest["environment"]["gpu_count"] != 0:
        raise ValueError("CPU qualification requires GPU visibility disabled")
    try:
        if scenario == "signal":

            def interrupt(signum, _frame):
                raise InterruptedError(f"qualification signal {signum}")

            previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            finally:
                signal.signal(signal.SIGTERM, previous)
        reference = context.array_store.write("qualification_values", np.arange(524288, dtype=np.float32))
        publish_metrics(
            context,
            dict(
                split="validation", summary_metrics={"qualification_loss": 0.0}, array_references=[reference.as_dict()]
            ),
        )
        if scenario == "array_tamper":
            path = context.paths.root / reference.path
            path.write_bytes(b"planned tamper")
            verify_array_reference(context.paths.root, reference)
    except BaseException:
        fail_run(context, exit_code=143 if scenario == "signal" else 1)
        raise
