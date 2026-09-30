"""Describe the machine a benchmark ran on."""

import datetime
import os
import platform
import socket
import subprocess
import sys


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _cpu_model():
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or None


def _memory_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _provider(explicit):
    if explicit:
        return explicit
    if any(k.startswith("VAST_") for k in os.environ):
        return "vast.ai"
    return None


def _git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    return _run(["git", "-C", here, "rev-parse", "--short", "HEAD"])


def _gpus():
    gpus = []
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        for d in wp.get_devices():
            if d.is_cuda:
                gpus.append({
                    "device": str(d),
                    "name": d.name,
                    "memory_gb": round(d.total_memory / 2**30, 1),
                })
    except Exception:
        pass
    smi = _run(["nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader"])
    if smi:
        for gpu, driver in zip(gpus, smi.splitlines()):
            gpu["driver"] = driver.strip()
    return gpus


def describe(label=None, provider=None):
    """Return a dict describing this machine and software environment."""
    import numpy
    import scipy

    import tmswarp

    try:
        import warp
        warp_version = warp.__version__
    except ImportError:
        warp_version = None

    return {
        "label": label,
        "provider": _provider(provider),
        "hostname": socket.gethostname(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()}",
        "os": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_threads": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "gpus": _gpus(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "warp": warp_version,
        "tmswarp": tmswarp.__version__,
        "tmswarp_commit": _git_commit(),
    }
