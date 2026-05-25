"""Regression test for the flashinfer per-call driver leak.

flashinfer's ``split_device_green_ctx`` leaks driver-level memory on
every call that is not recovered by ``cuStreamDestroy +
cuGreenCtxDestroy``, ``empty_cache``, or ``gc.collect``. This test acts
as a future-proofing guard: when flashinfer or NVIDIA fixes the upstream
issue this test will start failing, prompting us to relax the warning in
the user-facing docstrings.

We only run a handful of iterations (so the test stays fast) and assert
either:
  - leak is observable above a small tolerance (current behaviour), OR
  - leak is below the tolerance (upstream fixed — the test fails so we
    know to update the docs / status).

The assertion is structured as ``xfail``-style: we ``pytest.xfail`` when
leak is detected and ``fail`` if leak appears resolved.
"""

from __future__ import annotations

import gc
import subprocess

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="phyai.vgpu requires CUDA",
)


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


def _smi_mem_used_mib(device_idx: int = 0) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device_idx}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(out)


def test_flashinfer_split_leaks_or_upstream_fixed():
    if not _flashinfer_available():
        pytest.skip("flashinfer required")

    # nvidia-smi is the only window into driver-level memory; bail if it's
    # missing (e.g. inside a container without the binary).
    try:
        subprocess.check_output(["nvidia-smi", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("nvidia-smi not available")

    # Prime the primary context.
    _ = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()

    base_smi = _smi_mem_used_mib(0)

    iters = 5
    from flashinfer.green_ctx import split_device_green_ctx

    for _ in range(iters):
        streams, resources = split_device_green_ctx(
            torch.device("cuda:0"),
            2,
            16,
        )
        del streams, resources
        gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    end_smi = _smi_mem_used_mib(0)
    delta = end_smi - base_smi

    if delta >= 32:
        # Leak still present -> expected, document it explicitly.
        pytest.xfail(
            f"known flashinfer leak: 5 iter split_device_green_ctx "
            f"caused {delta} MiB driver-side growth. "
            f"vGPU must remain long-lived."
        )
    else:
        # If we ever land here, flashinfer or driver fixed it — the tests
        # fail loudly so we can update the warning text.
        pytest.fail(
            f"unexpected: only {delta} MiB driver growth across {iters} "
            f"split_device_green_ctx iterations. The upstream leak may "
            f"have been fixed — please update FlashInferBackend's "
            f"docstring."
        )
