#!/usr/bin/env python
"""
launch_tmux_ablation.py — batch-wise concurrent DAE ablation runs in tmux
(2025-06-21 histopathology edition)

Features:
• Run ablation studies sequentially with different configurations
• Use real histopathology data from HeparUnifiedPNG dataset
• Track metrics relevant to DAE models (PSNR, SSIM, etc.)
• Real-time progress monitoring with rich tables
• GPU resource tracking in tmux panes
• Summary statistics and best model selection
• WandB integration with online mode
• Data validation before training
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
import shlex
import shutil
import re
import datetime
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.errors import LiveError
from rich import box
import logging
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tmux_launcher.log')
    ]
)
logger = logging.getLogger("tmux_launcher")

# ── constants ───────────────────────────────────────────────────────────────
DEFAULT_BATCH = 2
REFRESH_RATE = 1
SESSION_BASE = "histo-ablation"
PY_MOD = "histopathology.src.training.dae_kan_attention.pl_training_robust"
REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = REPO_ROOT / "histopathology" / "data" / "processed" / "HeparUnifiedPNG"
WANDB_PROJECT = "histopathology-dae-kan"  # WandB project name
CUDA_MEMORY_CONFIG = "expandable_segments:True"  # CUDA memory config for PyTorch

console = Console()

# Metrics for the DAE model
VAL_METRICS = ["loss", "psnr", "ssim", "mae", "lpips"]
_REDUCE = {"loss": min, "lpips": min, **{m: max for m in VAL_METRICS if m not in ["loss", "lpips"]}}

_METRIC_RE = "|".join(VAL_METRICS)
_VAL_RE = re.compile(r"val[\\/_](?:val_)?(" + _METRIC_RE + r")[=:]\\s*([0-9.]+)")
_TRAIN_RE = re.compile(r"train[\\/_](?:train_)?(" + _METRIC_RE + r")[=:]\\s*([0-9.]+)")
_EPOCH_RE = re.compile(r"Epoch[\\s:]+(\\d+)(?:/(\\d+))?")
_ERR_RE = re.compile(r"Traceback \(most recent call last\):")

ICON_QUEUED, ICON_RUNNING, ICON_DONE, ICON_ERR = "⏳ queued", "🚀 running", "✅ done", "⚠ error"

# ── tmux helpers ────────────────────────────────────────────────────────────
def _tmux(*args: str | os.PathLike, check: bool = False) -> subprocess.CompletedProcess:
    """Run a tmux command with better error handling."""
    cmd = ["tmux", *map(str, args)]
    try:
        logger.debug(f"Running tmux command: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True, capture_output=True, check=check)
        if result.returncode != 0:
            logger.warning(f"Tmux command failed: {' '.join(cmd)}")
            logger.warning(f"Error: {result.stderr.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Tmux command failed with exit code {e.returncode}: {' '.join(cmd)}")
        logger.error(f"Error: {e.stderr.strip() if e.stderr else 'No error output'}")
        if check:
            raise
        return subprocess.CompletedProcess(cmd, e.returncode, "", str(e))
    except Exception as e:
        logger.error(f"Error executing tmux command: {e}")
        if check:
            raise
        return subprocess.CompletedProcess(cmd, 1, "", str(e))

def _list_job_windows(session: str) -> List[str]:
    """List all windows in a tmux session, with error handling."""
    try:
        out = _tmux("list-windows", "-t", session, "-F", "#{window_name}")
        if out.returncode == 0:
            windows = [w.strip() for w in out.stdout.splitlines() if w.strip()]
            return windows
        else:
            logger.warning(f"Failed to list windows for session {session}: {out.stderr.strip()}")
            return []
    except Exception as e:
        logger.error(f"Error listing windows for session {session}: {e}")
        return []

def _session_exists(session: str) -> bool:
    """Check if a tmux session exists."""
    result = _tmux("has-session", "-t", session)
    return result.returncode == 0

def _cleanup_session(session: str) -> None:
    """Clean up a tmux session and all its windows."""
    if _session_exists(session):
        logger.info(f"Cleaning up session: {session}")
        try:
            _tmux("kill-session", "-t", session)
        except Exception as e:
            logger.error(f"Error cleaning up session {session}: {e}")

# ── log helpers ─────────────────────────────────────────────────────────────
def _extract_best_metrics(log: Path) -> Dict[str, str]:
    best = {m: (float("inf") if m in ["loss", "lpips"] else float("-inf")) for m in VAL_METRICS}
    txt = log.read_text("utf-8", errors="ignore") if log.exists() else ""
    
    def upd(m, v): 
        best[m] = _REDUCE[m](best[m], float(v))
    
    for m, v in _VAL_RE.findall(txt):
        upd(m, float(v))
    for m, v in _TRAIN_RE.findall(txt):
        upd(m, float(v))
    
    out = {}
    for m in VAL_METRICS:
        if m in ["loss", "lpips"]:
            ok = best[m] < float("inf")
        else:
            ok = best[m] > float("-inf")
        out[f"val_{m}"] = f"{best[m]:.3f}" if ok else "—"
    
    return out

def _current_epoch(log: Path, max_epoch: int) -> str:
    if not log.exists(): 
        return "—"
    
    for line in reversed(log.read_text("utf-8", errors="ignore").splitlines()):
        if (m := _EPOCH_RE.search(line)):
            return f"{int(m.group(1))+1}/{m.group(2) or str(max_epoch)}"
    
    return "—"

_has_error = lambda log: log.exists() and bool(_ERR_RE.search(log.read_text("utf-8", errors="ignore")))

def _safe_float(value, default=0.0):
    """Safely convert a value to float, handling placeholders and errors"""
    if value == "—" or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# ── data validation utilities ────────────────────────────────────────────────
def _check_data_path(config_path: Path) -> bool:
    """Verify that the data path in the configuration exists and is valid."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check if dataset is specified and exists
        data_config = config.get('data', {})
        dataset_name = data_config.get('dataset', "HeparUnifiedPNG")
        dataset_path = Path(data_config.get('path', "histopathology/data/processed/HeparUnifiedPNG"))
        
        # Check if path is absolute
        if not dataset_path.is_absolute():
            dataset_path = REPO_ROOT / dataset_path
            
        if not dataset_path.exists():
            logger.warning(f"Dataset path {dataset_path} does not exist!")
            return False
            
        # Check for specific dataset files/structure
        if dataset_name == "HeparUnifiedPNG":
            # For HeparUnifiedPNG, check for tiles directory
            tiles_dir = dataset_path / "tiles"
            if not tiles_dir.exists():
                logger.warning(f"HeparUnifiedPNG tiles directory not found at {tiles_dir}")
                return False
                
            # Count images to ensure we have data
            image_count = len(list(tiles_dir.glob("*.png")))
            if image_count < 10:  # Arbitrary threshold
                logger.warning(f"Found only {image_count} images in {tiles_dir}. This seems too few.")
                return False
                
            logger.info(f"Found valid HeparUnifiedPNG dataset with {image_count} images")
            return True
        else:
            # For other datasets, just check the directory exists
            logger.info(f"Using dataset {dataset_name} at {dataset_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error validating data path: {e}")
        return False

# ── model utilities ─────────────────────────────────────────────────────────
def _param_size(config_path: Path) -> str:
    """Estimate the parameter size based on the ablation configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Base parameters estimate for DAE-KAN-Attention model
        base_params = 25.0  # Approximate base model size in millions
        
        # Adjust based on ablation configuration
        use_kan = config.get('model', {}).get('use_kan', True)
        use_eca = config.get('model', {}).get('use_eca', True)
        use_bam = config.get('model', {}).get('use_bam', True)
        
        # Apply rough adjustments based on components
        if not use_kan:
            base_params -= 8.5  # KAN layers contribute around 8.5M parameters
        if not use_eca:
            base_params -= 0.3  # ECA attention contributes around 0.3M parameters
        if not use_bam:
            base_params -= 1.2  # BAM attention contributes around 1.2M parameters
            
        return f"{base_params:.1f}M"
    except Exception as e:
        logger.error(f"Error estimating model size: {e}")
        return "—"

def _get_dataset_info(config_path: Path) -> str:
    """Extract dataset info from config for experiment naming."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        data_config = config.get('data', {})
        dataset_name = data_config.get('dataset', "HeparUnified")
        image_size = data_config.get('input_size', 128)
        
        return f"{dataset_name}-{image_size}px"
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return "unknown"

# ── table builder ───────────────────────────────────────────────────────────
def _make_table(ablations, params, exp_names, status, epochs, times, metrics) -> Table:
    tb = Table(title="Ablation Study Tracker (HeparUnifiedPNG)", box=box.SIMPLE_HEAVY, pad_edge=False)
    tb.add_column("#", justify="right", style="bold cyan")
    tb.add_column("Ablation", style="white")
    tb.add_column("Params", justify="right", style="blue")
    tb.add_column("Exp", style="bright_black")
    tb.add_column("Status", style="magenta")
    tb.add_column("Epoch", justify="center", style="yellow")
    tb.add_column("Time", justify="right", style="cyan")
    
    for m in VAL_METRICS:
        style = "red" if m in ["loss", "lpips"] else "green"
        tb.add_column(f"val_{m}", justify="right", style=style)
    
    for i, (a, p, e, s, ep, t, me) in enumerate(zip(ablations, params, exp_names, status, epochs, times, metrics), 1):
        tb.add_row(f"{i:>2}", a, p, e, s, ep, t, *[me.get(f"val_{k}", "—") for k in VAL_METRICS])
    
    return tb

# ── scheduler (runs one config) ─────────────────────────────────────────────
def _scheduler(ablations: List[str], params: List[str], configs: List[Path],
               batch: int, gpu: int, epoch: int, smoke_flag: str, exp_name: str, session: str):
    
    # Register cleanup function to ensure tmux sessions are cleaned up on exit
    atexit.register(_cleanup_session, session)
    
    # Register signal handlers for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, cleaning up...")
        _cleanup_session(session)
        # Restore original handlers and re-raise the signal
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        if sig == signal.SIGINT:
            raise KeyboardInterrupt
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    total = len(ablations)
    status = [ICON_QUEUED] * total
    epochs = ["—"] * total
    times = ["—"] * total
    metrics = [{f"val_{m}": "—" for m in VAL_METRICS} for _ in ablations]
    start_ts, end_ts = {}, {}
    exp_names = [exp_name] * total  # Create a list of experiment names
    active_jobs = set()  # Track active jobs

    def launch(idx: int):
        ablation_name = ablations[idx]
        logf = LOG_DIR / f"{ablation_name}.log"
        
        logger.info(f"Launching job {idx+1}/{total}: {ablation_name}")
        
        if logf.exists():
            try:
                logf.unlink()
            except OSError:
                backup_path = logf.with_suffix(f".old_{int(time.time())}")
                logger.info(f"Could not remove log file, backing up to {backup_path}")
                logf.rename(backup_path)
        
        # Set environment variables in a separate file to avoid shell parsing issues
        env_file = LOG_DIR / f"{ablation_name}_env.sh"
        with open(env_file, 'w') as f:
            f.write(f"export CUDA_VISIBLE_DEVICES={gpu}\n")
            f.write(f"export EXP_NAME={exp_name}_{ablation_name}\n")
            f.write(f"export PYTHONPATH={REPO_ROOT}\n")
            f.write(f"export WANDB_MODE=online\n")  # Enable online wandb sync
            f.write(f"export WANDB_PROJECT={WANDB_PROJECT}\n")  # Set WandB project
            f.write(f"export PYTORCH_CUDA_ALLOC_CONF={CUDA_MEMORY_CONFIG}\n")  # Memory optimization
                
        # Create a launch script to ensure proper execution
        script_file = LOG_DIR / f"{ablation_name}_launch.sh"
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"source {env_file}\n")
            f.write(f"cd {REPO_ROOT}\n")
            f.write(f"python -m {PY_MOD} --config {configs[idx]} {smoke_flag} 2>&1 | tee {logf}\n")
            
        # Make script executable
        os.chmod(script_file, 0o755)
        
        try:
            # Create the window with better error handling
            logger.debug(f"Creating tmux window for {ablation_name}")
            result = _tmux("new-window", "-d", "-t", session, "-n", ablation_name, 
                         str(script_file))
            
            if result.returncode != 0:
                logger.error(f"Failed to create tmux window for {ablation_name}: {result.stderr}")
                status[idx] = ICON_ERR
                return False
                
            active_jobs.add(ablation_name)
            start_ts[idx] = time.time()
            return True
        except Exception as e:
            logger.error(f"Error launching job {ablation_name}: {e}")
            status[idx] = ICON_ERR
            return False

    fmt = lambda s: str(datetime.timedelta(seconds=int(s)))
    
    def running_windows():
        windows = _list_job_windows(session)
        return [w for w in windows if w != "dashboard"]

    # Initialize the live display
    live = None
    if console.is_terminal:
        try:
            live = Live(console=console, refresh_per_second=5)
            live.__enter__()
            logger.info("Initialized live display")
        except LiveError as e:
            logger.warning(f"Could not initialize live display: {e}")
            live = None

    try:
        launched = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        logger.info(f"Starting scheduler with {total} jobs, batch size {batch}")
        
        while launched < total or running_windows():
            try:
                # Get current running windows
                running = set(running_windows())
                logger.debug(f"Running windows: {running}")
                
                # Launch new jobs as needed
                free = batch - len(running)
                while free > 0 and launched < total:
                    if launch(launched):
                        status[launched] = ICON_RUNNING
                        consecutive_failures = 0
                        logger.info(f"Job {launched+1}/{total} ({ablations[launched]}) started")
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Failed to launch job {launched+1}/{total} ({ablations[launched]})")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"Too many consecutive failures ({consecutive_failures}), aborting")
                            raise RuntimeError("Too many consecutive job launch failures")
                    
                    free -= 1
                    launched += 1

                # Update job status
                now = time.time()
                
                for i, ablation_name in enumerate(ablations):
                    logf = LOG_DIR / f"{ablation_name}.log"
                    
                    # Update running jobs
                    if status[i] == ICON_RUNNING:
                        epochs[i] = _current_epoch(logf, epoch)
                        metrics[i] = _extract_best_metrics(logf)
                        times[i] = fmt(now - start_ts.get(i, now))
                        
                    # Check if a running job has completed
                    if status[i] == ICON_RUNNING and ablation_name not in running:
                        logger.info(f"Job {ablation_name} has completed")
                        status[i] = ICON_ERR if _has_error(logf) else ICON_DONE
                        metrics[i] = _extract_best_metrics(logf)
                        times[i] = fmt(now - start_ts.get(i, now))
                        end_ts[i] = time.time()
                        
                        # Clean up the window
                        try:
                            _tmux("kill-window", "-t", f"{session}:{ablation_name}")
                            active_jobs.discard(ablation_name)
                        except Exception as e:
                            logger.warning(f"Error cleaning up window for {ablation_name}: {e}")

                # Build and display the status table
                tbl = _make_table(ablations, params, exp_names, status, epochs, times, metrics)
                
                # Add summary row if any jobs have completed
                if end_ts:
                    tot = fmt(max(end_ts.values()) - min(start_ts.values()))
                    tbl.add_section()
                    tbl.add_row("", "", "", "", "", "", "⏱ " + tot, *[""] * len(VAL_METRICS))
                    
                # Update the display
                if live:
                    live.update(tbl)
                else:
                    console.clear()
                    console.print(tbl)
                    
                # Sleep before next update
                time.sleep(REFRESH_RATE)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, cleaning up...")
                raise
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                if len(running_windows()) == 0 and launched >= total:
                    logger.info("All jobs appear to be done despite error, continuing...")
                    break
                
                # Sleep a bit longer after errors to avoid thrashing
                time.sleep(REFRESH_RATE * 2)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, cleaning up...")
        # Cleanup is handled by the atexit handler
    except Exception as e:
        logger.error(f"Fatal error in scheduler: {e}")
    finally:
        # Clean up the live display
        if live:
            try:
                live.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error cleaning up live display: {e}")
                
        # Clean up any remaining jobs
        for name in list(active_jobs):
            try:
                logger.info(f"Cleaning up job {name}")
                _tmux("kill-window", "-t", f"{session}:{name}")
            except Exception as e:
                logger.warning(f"Error cleaning up job {name}: {e}")

    # Final summary
    if end_ts:
        tot = max(end_ts.values()) - min(start_ts.values())
        mean = tot / total
        
        # Safely gather PSNR scores for completed runs
        psnr_scores = []
        for i in range(total):
            psnr_value = _safe_float(metrics[i].get("val_psnr", "—"), default=-float("inf"))
            if psnr_value > -float("inf"):  # Only include valid scores
                psnr_scores.append((i, psnr_value))
        
        # Only report best model if we have valid scores
        best_model_info = ""
        if psnr_scores:
            best_i, best_psnr = max(psnr_scores, key=lambda x: x[1])
            best_model_info = f"  •  Best model: [bold]{ablations[best_i]}[/] (val_psnr={best_psnr:.3f})"
        
        console.rule("[green]Ablation study finished")
        console.print(f"Total wall-time: {fmt(tot)}  •  Mean/job: {fmt(mean)}{best_model_info}")

# ── CLI ─────────────────────────────────────────────────────────────────────
@click.command(context_settings={"show_default": True})
@click.option("--base-config", "-c", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default="histopathology/configs/wandb_config.yaml",
              help="Base YAML config to use with ablations")
@click.option("--ablation-dir", "-a", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default="histopathology/configs/ablations",
              help="Directory containing ablation config files")
@click.option("--batch", "-b", type=int, default=DEFAULT_BATCH, help="Max concurrent tmux panes")
@click.option("--smoke", is_flag=True, help="Pass --smoke to every child run (fast_dev_run)")
@click.option("--name", "-n", default="hepar", help="Experiment tag (shown & in ENV EXP_NAME)")
@click.option("--exclude", "-ex", multiple=True, help="Excluding ablation configs to be trained")
@click.option("--gpu", "-g", type=int, default=0, help="Force a single GPU index")
@click.option("--epochs", "-e", type=int, default=30, help="Override max epochs")
@click.option("--wandb-offline", is_flag=True, help="Run WandB in offline mode")
@click.option("--validate-data", is_flag=True, default=True, help="Validate dataset existence before training")
def cli(base_config: Path, ablation_dir: Path, batch: int, smoke: bool, name: str,
        exclude: List[str], gpu: int, epochs: int, wandb_offline: bool, validate_data: bool):
    
    logger.info(f"Starting ablation runner with: base_config={base_config}, batch={batch}, gpu={gpu}, epochs={epochs}, dataset=HeparUnifiedPNG")
    
    # Check for tmux
    if not shutil.which("tmux"):
        error_msg = "tmux not found. Please install tmux to use this script."
        logger.error(error_msg)
        console.print(f"[red]❌ {error_msg}")
        sys.exit(1)
        
    # Validate the data path first
    if validate_data:
        console.print("[bold]Validating HeparUnifiedPNG dataset...[/]")
        if not DATA_DIR.exists():
            error_msg = f"HeparUnifiedPNG dataset path not found: {DATA_DIR}"
            logger.error(error_msg)
            console.print(f"[red]❌ {error_msg}")
            console.print("[yellow]Please ensure the HeparUnifiedPNG dataset is in histopathology/data/processed/HeparUnifiedPNG[/]")
            sys.exit(1)
            
        # Check for image files
        tiles_dir = DATA_DIR / "tiles"
        if not tiles_dir.exists():
            error_msg = f"HeparUnifiedPNG tiles directory not found: {tiles_dir}"
            logger.error(error_msg)
            console.print(f"[red]❌ {error_msg}")
            sys.exit(1)
            
        image_count = len(list(tiles_dir.glob("*.png")))
        if image_count < 10:
            error_msg = f"Found only {image_count} images in HeparUnifiedPNG dataset. This seems too few."
            logger.warning(error_msg)
            console.print(f"[yellow]⚠ {error_msg}")
            continue_anyway = click.confirm("Continue anyway?", default=False)
            if not continue_anyway:
                sys.exit(1)
        else:
            console.print(f"[green]✓ Found {image_count} images in HeparUnifiedPNG dataset[/]")

    # Find all ablation configs
    try:
        ablation_files = glob.glob(f"{ablation_dir}/*.yaml")
        if not ablation_files:
            error_msg = f"No ablation config files found in {ablation_dir}"
            logger.error(error_msg)
            console.print(f"[red]❌ {error_msg}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error finding ablation configs: {e}")
        console.print(f"[red]❌ Error finding ablation configs: {e}")
        sys.exit(1)
        
    # Set WandB mode based on flag
    global WANDB_MODE
    WANDB_MODE = "offline" if wandb_offline else "online"
    logger.info(f"WandB mode: {WANDB_MODE}")

    smoke_flag = "--smoke" if smoke else ""
    logger.info(f"Smoke test: {'enabled' if smoke else 'disabled'}")

    # Get ablation config names without extensions
    ablations = [Path(f).stem for f in ablation_files if Path(f).stem not in exclude]
    if not ablations:
        error_msg = "No valid ablation configs after exclusions"
        logger.error(error_msg)
        console.print(f"[red]❌ {error_msg}")
        sys.exit(1)

    # Full paths to config files
    configs = [Path(f) for f in ablation_files if Path(f).stem in ablations]
    logger.info(f"Found {len(configs)} ablation configs: {', '.join(ablations)}")
    
    # Validate configs have proper data paths
    if validate_data:
        console.print("[bold]Validating data paths in configs...")
        invalid_configs = []
        for i, config in enumerate(configs):
            if not _check_data_path(config):
                console.print(f"[yellow]⚠ Config {ablations[i]} has invalid data path[/]")
                invalid_configs.append(ablations[i])
                
        if invalid_configs:
            error_msg = f"The following configs have invalid data paths: {', '.join(invalid_configs)}"
            logger.warning(error_msg)
            console.print(f"[yellow]⚠ {error_msg}")
            continue_anyway = click.confirm("Continue anyway?", default=False)
            if not continue_anyway:
                sys.exit(1)
    
    # Calculate parameter counts for each config
    console.print("[bold]Estimating model parameters...")
    params = []
    for config in configs:
        try:
            params.append(_param_size(config))
        except Exception as e:
            logger.warning(f"Error estimating parameters for {config}: {e}")
            params.append("—")
            
    # Get dataset info for experiment naming
    dataset_info = "HeparUnifiedPNG"
    exp_name_base = f"{name}-{dataset_info}"

    # Create unique session name
    session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session = f"{name}_{SESSION_BASE}_{session_id}"
    logger.info(f"Creating tmux session: {session}")
    
    # Clean up any existing session with the same name
    if _session_exists(session):
        logger.info(f"Session {session} already exists, killing it")
        _tmux("kill-session", "-t", session)
    
    # Create the session
    try:
        logger.info("Creating new tmux session")
        result = _tmux("new-session", "-d", "-s", session, "-n", "dashboard", "watch -n1 nvidia-smi")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create tmux session: {result.stderr}")
            
        # Configure session options
        _tmux("set-option", "-t", session, "-g", "remain-on-exit", "off")
        _tmux("set-option", "-t", session, "-g", "history-limit", "5000")
        
        logger.info(f"Session {session} created successfully")
    except Exception as e:
        logger.error(f"Error creating tmux session: {e}")
        console.print(f"[red]❌ Error creating tmux session: {e}")
        sys.exit(1)
    
    console.rule(f"🚀 Launching {len(ablations)} ablation runs with HeparUnifiedPNG data")
    console.print(f"Base config: [green]{base_config.name}[/]")
    console.print(f"Dataset: [blue]HeparUnifiedPNG[/] (in {DATA_DIR})")
    console.print(f"Ablations: [cyan]{', '.join(ablations)}[/]")
    console.print(f"Session: [yellow]{session}[/] (attach with: tmux attach -t {session})")
    console.print(f"WandB: [{'green' if WANDB_MODE == 'online' else 'yellow'}]{WANDB_MODE}[/] mode")

    try:
        # Register a cleanup function for this specific session
        atexit.register(_cleanup_session, session)
        
        # Run the scheduler
        _scheduler(ablations, params, configs, batch, gpu, epochs, smoke_flag, exp_name_base, session)
        
        # If we get here, everything completed successfully
        logger.info("All jobs completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        console.print("\n[red]Interrupted — tearing down tmux session...")
        _cleanup_session(session)
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error in scheduler: {e}")
        console.print(f"\n[red]Error in scheduler: {e}")
        _cleanup_session(session)
        sys.exit(1)
    finally:
        # Unregister the cleanup function if we're exiting normally
        atexit.unregister(_cleanup_session)

if __name__ == "__main__":
    cli()
