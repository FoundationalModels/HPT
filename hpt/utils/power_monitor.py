"""
Tracks GPU and CPU power draw during training, accumulating energy (Wh).

GPU: pynvml (nvidia-ml-py) — works without root on most clusters.
CPU: Linux RAPL via /sys/class/powercap/ — silently skipped if inaccessible.

Usage:
    monitor = PowerMonitor(gpu_indices=[0], sample_interval=1.0)
    monitor.start()
    ...
    stats = monitor.read_and_reset()   # per-epoch snapshot
    monitor.stop()
    cumulative = monitor.cumulative_stats()
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# GPU via pynvml
# ---------------------------------------------------------------------------

def _init_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _gpu_handles(pynvml, gpu_indices: Optional[List[int]]):
    count = pynvml.nvmlDeviceGetCount()
    indices = gpu_indices if gpu_indices is not None else list(range(count))
    return {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices if i < count}


# ---------------------------------------------------------------------------
# CPU via RAPL
# ---------------------------------------------------------------------------

_RAPL_ROOT = Path("/sys/class/powercap")

def _rapl_domains() -> Dict[str, Path]:
    """Return {domain_name: energy_uj_path} for readable RAPL domains."""
    domains = {}
    if not _RAPL_ROOT.exists():
        return domains
    for entry in sorted(_RAPL_ROOT.iterdir()):
        energy_path = entry / "energy_uj"
        name_path = entry / "name"
        if not energy_path.exists() or not name_path.exists():
            continue
        try:
            energy_path.read_text()          # test readability
            name = name_path.read_text().strip()
            domains[name] = energy_path
        except PermissionError:
            pass
    return domains


def _read_energy_uj(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Accumulated stats
# ---------------------------------------------------------------------------

@dataclass
class _AccStats:
    """Accumulates power samples into energy and running average."""
    energy_j: float = 0.0          # total joules
    power_sum: float = 0.0         # sum of sampled watts (for mean)
    sample_count: int = 0
    peak_watts: float = 0.0

    def add(self, watts: float, dt: float):
        self.energy_j += watts * dt
        self.power_sum += watts
        self.sample_count += 1
        if watts > self.peak_watts:
            self.peak_watts = watts

    @property
    def avg_watts(self) -> float:
        return self.power_sum / self.sample_count if self.sample_count else 0.0

    @property
    def energy_wh(self) -> float:
        return self.energy_j / 3600.0

    def snapshot_and_reset(self) -> dict:
        """Return a stats dict then reset accumulators (but keep peak)."""
        out = {
            "energy_wh": self.energy_wh,
            "avg_watts": self.avg_watts,
            "peak_watts": self.peak_watts,
        }
        self.energy_j = 0.0
        self.power_sum = 0.0
        self.sample_count = 0
        # intentionally keep peak_watts across resets for the full run
        return out


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class PowerMonitor:
    """
    Background-thread power monitor for GPU (pynvml) and CPU (RAPL).

    Args:
        gpu_indices: which GPU device indices to monitor (None = all).
        sample_interval: seconds between samples.
    """

    def __init__(self, gpu_indices: Optional[List[int]] = None, sample_interval: float = 1.0):
        self._interval = sample_interval
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # GPU setup
        self._pynvml = _init_nvml()
        self._gpu_handles: Dict[int, object] = {}
        if self._pynvml:
            self._gpu_handles = _gpu_handles(self._pynvml, gpu_indices)

        # CPU setup
        self._rapl_domains = _rapl_domains()
        self._rapl_prev: Dict[str, int] = {}   # last energy_uj reading

        # Accumulators: per-GPU index and per-RAPL domain, plus total GPU
        self._gpu_stats: Dict[int, _AccStats] = {i: _AccStats() for i in self._gpu_handles}
        self._cpu_stats: Dict[str, _AccStats] = {n: _AccStats() for n in self._rapl_domains}

        # Cumulative mirrors (never reset, for full-run totals)
        self._gpu_total: Dict[int, _AccStats] = {i: _AccStats() for i in self._gpu_handles}
        self._cpu_total: Dict[str, _AccStats] = {n: _AccStats() for n in self._rapl_domains}

        if not self._pynvml and not self._rapl_domains:
            print("[power_monitor] WARNING: neither pynvml nor RAPL is available — power tracking disabled")
        elif not self._pynvml:
            print("[power_monitor] pynvml unavailable; GPU power tracking disabled")
        elif not self._rapl_domains:
            print("[power_monitor] RAPL unavailable (likely needs root); CPU power tracking disabled")

    # ------------------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        # Prime RAPL baseline
        for name, path in self._rapl_domains.items():
            v = _read_energy_uj(path)
            if v is not None:
                self._rapl_prev[name] = v
        self._thread = threading.Thread(target=self._run, daemon=True, name="power_monitor")
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval * 3)
        self._thread = None
        if self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------

    def _run(self):
        prev_time = time.monotonic()
        while not self._stop_event.wait(timeout=self._interval):
            now = time.monotonic()
            dt = now - prev_time
            prev_time = now
            self._sample(dt)

    def _sample(self, dt: float):
        with self._lock:
            # GPU
            for idx, handle in self._gpu_handles.items():
                try:
                    mw = self._pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
                    watts = mw / 1000.0
                    self._gpu_stats[idx].add(watts, dt)
                    self._gpu_total[idx].add(watts, dt)
                except Exception:
                    pass

            # CPU via RAPL (delta energy / dt → watts)
            for name, path in self._rapl_domains.items():
                v = _read_energy_uj(path)
                if v is None:
                    continue
                prev = self._rapl_prev.get(name)
                if prev is not None:
                    delta_uj = v - prev
                    # Handle wraparound
                    max_range_path = path.parent / "max_energy_range_uj"
                    if delta_uj < 0:
                        try:
                            max_uj = int(max_range_path.read_text().strip())
                            delta_uj += max_uj
                        except Exception:
                            delta_uj = 0
                    watts = (delta_uj / 1e6) / dt if dt > 0 else 0.0
                    self._cpu_stats[name].add(watts, dt)
                    self._cpu_total[name].add(watts, dt)
                self._rapl_prev[name] = v

    # ------------------------------------------------------------------

    def read_and_reset(self) -> dict:
        """
        Return stats accumulated since last call (or since start), then reset.
        Suitable for per-epoch logging.
        """
        with self._lock:
            out = {}
            for idx, acc in self._gpu_stats.items():
                out[f"gpu{idx}/power_avg_w"] = acc.avg_watts
                out[f"gpu{idx}/power_peak_w"] = acc.peak_watts
                out[f"gpu{idx}/energy_wh"] = acc.energy_wh
                acc.snapshot_and_reset()
            for name, acc in self._cpu_stats.items():
                key = name.replace("-", "_")
                out[f"cpu_{key}/power_avg_w"] = acc.avg_watts
                out[f"cpu_{key}/energy_wh"] = acc.energy_wh
                acc.snapshot_and_reset()
        return out

    def cumulative_stats(self) -> dict:
        """Return totals over the entire monitored period (never reset)."""
        with self._lock:
            out = {}
            for idx, acc in self._gpu_total.items():
                out[f"gpu{idx}/total_energy_wh"] = acc.energy_wh
                out[f"gpu{idx}/overall_avg_w"] = acc.avg_watts
                out[f"gpu{idx}/peak_w"] = acc.peak_watts
            for name, acc in self._cpu_total.items():
                key = name.replace("-", "_")
                out[f"cpu_{key}/total_energy_wh"] = acc.energy_wh
                out[f"cpu_{key}/overall_avg_w"] = acc.avg_watts
        return out

    def summary_str(self) -> str:
        stats = self.cumulative_stats()
        lines = ["[power_monitor] Training energy summary:"]
        for k, v in sorted(stats.items()):
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)
