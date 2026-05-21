import atexit
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import time

from .config import config



def get_optimal_device() -> str:
    """Determine optimal compute device (MPS, CUDA, or CPU)."""
    if config.device != "auto":
        return config.device

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"

    # We could check torch.cuda.is_available() but avoiding importing torch eagerly
    # by default. A more robust check could be added here.
    return "cpu"


def setup_cache():
    """Ensure cache directory exists."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)


def get_total_memory() -> int:
    """Get total system memory in bytes."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return page_size * phys_pages
    except Exception:
        pass

    # Fallback to sysctl on macOS
    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except Exception:
        pass

    return 8 * 1024 * 1024 * 1024  # default to 8GB fallback


def setup_memory_guards():
    """Apply memory limits to PyTorch MPS and MLX backends to safeguard system resources."""
    fraction = config.max_memory_fraction
    if fraction <= 0.0 or fraction > 1.0:
        return

    # 1. PyTorch MPS limit
    try:
        import torch

        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.set_per_process_memory_fraction(fraction)
    except ImportError:
        pass
    except Exception:
        pass

    # 2. MLX limit
    try:
        import mlx.core as mx

        total_mem = get_total_memory()
        limit_bytes = int(total_mem * fraction)
        
        # Try mx.set_memory_limit, fallback to mx.metal.set_memory_limit
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(limit_bytes)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "set_memory_limit"):
            mx.metal.set_memory_limit(limit_bytes)

        # Cap the compile/scratch cache to 10% of the memory limit
        cache_limit_bytes = int(limit_bytes * 0.1)
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(cache_limit_bytes)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(cache_limit_bytes)
    except ImportError:
        pass
    except Exception:
        pass


def cleanup_memory(force_sys_purge: bool = False):
    """Clean up memory across PyTorch, MLX, and Python garbage collection.

    This function checks if torch or mlx are imported/available and releases
    their respective caches to return memory back to the OS. It also sweeps
    orphaned background worker processes.
    """
    # 1. Sweep orphaned processes first
    reap_orphaned_processes()

    # 2. PyTorch MPS
    if "torch" in sys.modules:
        try:
            import torch

            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # 3. MLX
    if "mlx.core" in sys.modules or "mlx" in sys.modules:
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            pass

    # 4. Python Garbage Collection
    import gc

    gc.collect()

    # 5. OS-level VM page purge (optional and non-blocking)
    if force_sys_purge and platform.system() == "Darwin":
        try:
            # Run sudo purge in non-interactive mode. Will only succeed
            # if user configured passwordless sudo for the purge command.
            subprocess.run(["sudo", "-n", "purge"], capture_output=True, check=False)
        except Exception:
            pass


def reap_orphaned_processes():
    """Identify and terminate orphaned AIServices worker processes (PPID = 1).

    Detects orphans by matching against the virtual-environment path of the
    currently-running interpreter, so the check is portable across different
    project directory names.  The function is a no-op on Windows where the
    ``ps`` POSIX command is unavailable.
    """
    # ps -eo is a POSIX/macOS/Linux tool; skip on Windows
    if platform.system() == "Windows":
        return

    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return

        # Derive the venv prefix from the running interpreter so the match
        # is independent of the project directory name.
        venv_prefix = os.path.dirname(os.path.dirname(sys.executable))
        current_pid = os.getpid()

        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid_str, ppid_str, command = parts
            try:
                pid = int(pid_str)
                ppid = int(ppid_str)
            except ValueError:
                continue

            # PPID = 1 → adopted by init/launchd → orphaned worker
            if ppid == 1 and pid != current_pid:
                if venv_prefix in command and "python" in command:
                    try:
                        # Attempt graceful shutdown first; escalate to SIGKILL
                        # only if the process is still alive after 0.5 s.
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)
                        os.kill(pid, signal.SIGKILL)  # no-op if already dead
                    except ProcessLookupError:
                        pass  # already gone — nothing to do
                    except PermissionError:
                        pass  # not our process to kill
    except Exception:
        pass


def terminate_current_children():
    """Terminate all active child processes spawned by the current process."""
    try:
        for p in multiprocessing.active_children():
            try:
                p.terminate()
                p.join(timeout=0.5)
                if p.is_alive():
                    p.kill()
            except Exception:
                pass
    except Exception:
        pass


# Register atexit handler to ensure automatic cleanup of children and orphans on exit
def _exit_handler():
    terminate_current_children()
    reap_orphaned_processes()


atexit.register(_exit_handler)

