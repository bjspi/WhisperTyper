"""List the devices the installed backends can use; prints one JSON object.

Runs as its own process (``python -m app.local_engine.devices``), so the app never loads
OpenVINO or CUDA itself: ``{"devices": [{"id": "cuda", "name": "..."}, ...], "errors": {...}}``.
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

from app.local_engine.availability import display_adapters


def detect() -> Dict[str, object]:
    """The usable devices and the errors of backends that could not be checked."""
    devices: List[Dict[str, str]] = []
    errors: Dict[str, str] = {}
    adapters = display_adapters()
    try:
        from app.local_engine.backends import add_nvidia_dll_dirs

        add_nvidia_dll_dirs()
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            nvidia = next((name for name in adapters if "nvidia" in name.lower()), "NVIDIA GPU")
            devices.append({"id": "cuda", "name": nvidia})
        cpu = True
    except Exception as exc:  # noqa: BLE001 - reported to the settings page
        errors["ctranslate2"] = f"{type(exc).__name__}: {exc}"
        cpu = False
    try:
        from openvino import Core

        core = Core()
        for ov_device, device_id in (("GPU", "intel_gpu"), ("NPU", "intel_npu"), ("CPU", "openvino_cpu")):
            if ov_device in core.available_devices:
                devices.append({"id": device_id, "name": str(core.get_property(ov_device, "FULL_DEVICE_NAME"))})
    except Exception as exc:  # noqa: BLE001
        errors["openvino"] = f"{type(exc).__name__}: {exc}"
    if cpu:
        import platform

        ov_cpu = next((device["name"] for device in devices if device["id"] == "openvino_cpu"), "")
        devices.append({"id": "cpu", "name": ov_cpu or platform.processor() or "CPU"})
    return {"devices": devices, "errors": errors}


def main() -> int:
    """Print the detection result as JSON."""
    print(json.dumps(detect(), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
