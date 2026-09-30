"""
env_info.py
-------------
Captures and reports the hardware and software environment every
timing measurement in this study depends on, directly addressing the
reviewer's point that none of this was previously reported: "Report
the hardware (CPU, RAM, OS), Python, NumPy and pandas versions, the
number of repeats, and the variance for all tables."

Run this once at the start of a benchmarking session and save its
output alongside your results, since runtime comparisons are only
meaningful in the context of the specific machine they were measured
on.

Run:
    python env_info.py
"""
import json
import platform
import subprocess
import sys


def get_cpu_info():
    """Best-effort CPU model string; falls back gracefully across
    platforms rather than failing if a given method isn't available."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            return out.strip()
        elif system == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return lines[1] if len(lines) > 1 else "unknown"
    except Exception as e:
        return f"could not determine ({e})"
    return "unknown"


def get_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        return "unknown (install psutil for this: pip install psutil)"


def get_package_versions():
    versions = {}
    for pkg in ["numpy", "pandas", "mlxtend", "scipy", "matplotlib"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def collect_env_info():
    info = dict(
        cpu=get_cpu_info(),
        cpu_count_logical=None,
        ram_gb=get_ram_gb(),
        os=f"{platform.system()} {platform.release()}",
        os_full=platform.platform(),
        python_version=sys.version.split()[0],
        packages=get_package_versions(),
    )
    try:
        import os as os_module
        info["cpu_count_logical"] = os_module.cpu_count()
    except Exception:
        pass
    return info


if __name__ == "__main__":
    info = collect_env_info()
    print("Environment used for this study's timing measurements:")
    print(f"  CPU: {info['cpu']} ({info['cpu_count_logical']} logical cores)")
    print(f"  RAM: {info['ram_gb']} GB")
    print(f"  OS: {info['os_full']}")
    print(f"  Python: {info['python_version']}")
    print(f"  Package versions:")
    for pkg, ver in info["packages"].items():
        print(f"    {pkg}: {ver}")

    with open("env_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved to env_info.json -- include this alongside any reported timing "
          f"results, since runtime comparisons are only meaningful in the context "
          f"of the specific machine they were measured on.")
