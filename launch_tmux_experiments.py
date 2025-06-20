#!/usr/bin/env python
"""
launch_tmux_experiments.py — Run multiple experiments sequentially with tmux
(2025-06-21 histopathology edition)

Features:
• Run different experiments from experiments.yaml 
• Combine with ablation studies
• Real-time progress monitoring with rich tables
• GPU resource tracking in tmux panes
• Summary statistics and best model selection
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
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import click
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.errors import LiveError
from rich import box

# ── constants ───────────────────────────────────────────────────────────────
DEFAULT_BATCH = 2
REFRESH_RATE = 1
SESSION_BASE = "histo-experiments"
PY_MOD = "histopathology.src.training.dae_kan_attention.pl_training_robust"
REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

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
def _tmux(*args: str | os.PathLike):
    return subprocess.run(["tmux", *map(str, args)], text=True, capture_output=True)

def _list_job_windows(session: str) -> List[str]:
    out = _tmux("list-windows", "-t", session, "-F", "#{window_name}")
    return [w.strip() for w in out.stdout.splitlines() if w.strip()] if out.returncode == 0 else []

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

# ── model utilities ─────────────────────────────────────────────────────────
def _param_size(config: Dict) -> str:
    """Estimate the parameter size based on the model configuration"""
    try:
        # Base parameters estimate for DAE-KAN-Attention model
        base_params = 25.0  # Approximate base model size in millions
        
        # Adjust based on model configuration
        model_config = config.get('model', {})
        use_kan = model_config.get('use_kan', True)
        use_eca = model_config.get('use_eca', True)
        use_bam = model_config.get('use_bam', True)
        high_capacity = model_config.get('high_capacity', False)
        low_capacity = model_config.get('low_capacity', False)
        
        # Apply rough adjustments based on components
        if not use_kan:
            base_params -= 8.5  # KAN layers contribute around 8.5M parameters
        if not use_eca:
            base_params -= 0.3  # ECA attention contributes around 0.3M parameters
        if not use_bam:
            base_params -= 1.2  # BAM attention contributes around 1.2M parameters
            
        # Adjust for capacity variations
        if high_capacity:
            base_params *= 1.5
        if low_capacity:
            base_params *= 0.7
            
        return f"{base_params:.1f}M"
    except Exception as e:
        console.print(f"[yellow]Error estimating model size: {e}")
        return "—"

# ── table builder ───────────────────────────────────────────────────────────
def _make_table(experiment_names, params, statuses, epochs, times, metrics) -> Table:
    tb = Table(title="Experiment Tracker", box=box.SIMPLE_HEAVY, pad_edge=False)
    tb.add_column("#", justify="right", style="bold cyan")
    tb.add_column("Experiment", style="white")
    tb.add_column("Params", justify="right", style="blue")
    tb.add_column("Status", style="magenta")
    tb.add_column("Epoch", justify="center", style="yellow")
    tb.add_column("Time", justify="right", style="cyan")
    
    for m in VAL_METRICS:
        style = "red" if m in ["loss", "lpips"] else "green"
        tb.add_column(f"val_{m}", justify="right", style=style)
    
    for i, (exp, p, s, ep, t, me) in enumerate(zip(experiment_names, params, statuses, epochs, times, metrics), 1):
        tb.add_row(f"{i:>2}", exp, p, s, ep, t, *[me.get(f"val_{k}", "—") for k in VAL_METRICS])
    
    return tb

# ── scheduler (runs experiments) ─────────────────────────────────────────────
def _scheduler(experiment_names: List[str], config_paths: List[Path], params: List[str],
               batch: int, gpu: int, smoke_flag: str, session: str):
    
    total = len(experiment_names)
    statuses = [ICON_QUEUED] * total
    epochs = ["—"] * total
    times = ["—"] * total
    metrics = [{f"val_{m}": "—" for m in VAL_METRICS} for _ in experiment_names]
    start_ts, end_ts = {}, {}

    def launch(idx: int):
        exp_name = experiment_names[idx]
        logf = LOG_DIR / f"{exp_name}.log"
        
        if logf.exists():
            try:
                logf.unlink()
            except OSError:
                logf.rename(logf.with_suffix(".old"))
                
        cmd = (f"CUDA_VISIBLE_DEVICES={gpu} EXP_NAME={exp_name} PYTHONPATH={REPO_ROOT} "
               f"python -m {PY_MOD} --config {shlex.quote(str(config_paths[idx]))} {smoke_flag} "
               f"|& tee {shlex.quote(str(logf))}")
               
        _tmux("new-window", "-t", session, "-n", exp_name, "-c", str(REPO_ROOT), 
              f"bash -lc {shlex.quote(cmd)}")
              
        start_ts[idx] = time.time()

    fmt = lambda s: str(datetime.timedelta(seconds=int(s)))
    running_windows = lambda: [w for w in _list_job_windows(session) if w != "dashboard"]

    live = Live(console=console, refresh_per_second=10) if console.is_terminal else None
    if live:
        try:
            live.__enter__()
        except LiveError:
            live = None

    try:
        launched = 0
        while launched < total or running_windows():
            free = batch - len(running_windows())
            while free > 0 and launched < total:
                launch(launched)
                statuses[launched] = ICON_RUNNING
                free -= 1
                launched += 1

            now = time.time()
            running = set(running_windows())
            
            for i, exp_name in enumerate(experiment_names):
                logf = LOG_DIR / f"{exp_name}.log"
                
                if statuses[i] == ICON_RUNNING:
                    # Determine max epochs from config
                    config = OmegaConf.load(config_paths[i])
                    max_epochs = config.get('training', {}).get('max_epochs', 100)
                    
                    epochs[i] = _current_epoch(logf, max_epochs)
                    metrics[i] = _extract_best_metrics(logf)
                    times[i] = fmt(now - start_ts.get(i, now))
                    
                if statuses[i] == ICON_RUNNING and exp_name not in running:
                    statuses[i] = ICON_ERR if _has_error(logf) else ICON_DONE
                    metrics[i] = _extract_best_metrics(logf)
                    times[i] = fmt(now - start_ts.get(i, now))
                    end_ts[i] = time.time()
                    _tmux("kill-window", "-t", f"{session}:{exp_name}")

            tbl = _make_table(experiment_names, params, statuses, epochs, times, metrics)
            
            # Live summary row
            if end_ts:
                tot = fmt(max(end_ts.values()) - min(start_ts.values()))
                tbl.add_section()
                tbl.add_row("", "", "", "", "", "⏱ " + tot, *[""] * len(VAL_METRICS))
                
            if live:
                live.update(tbl)
            else:
                console.clear()
                console.print(tbl)
                
            time.sleep(REFRESH_RATE)
                
    finally:
        if live:
            live.__exit__(None, None, None)

    # Final summary
    if end_ts:
        tot = max(end_ts.values()) - min(start_ts.values())
        mean = tot / total
        psnr_scores = [(i, float(metrics[i].get("val_psnr", 0))) for i in range(total)]
        best_i, best_psnr = max(psnr_scores, key=lambda x: x[1])
        
        console.rule("[green]Experiments finished")
        console.print(f"Total wall-time: {fmt(tot)}  •  Mean/job: {fmt(mean)}  •  "
                      f"Best experiment: [bold]{experiment_names[best_i]}[/] (val_psnr={best_psnr:.3f})")

# ── CLI ─────────────────────────────────────────────────────────────────────
@click.command(context_settings={"show_default": True})
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default="histopathology/configs/experiments.yaml",
              help="Experiments configuration file")
@click.option("--batch", "-b", type=int, default=DEFAULT_BATCH, help="Max concurrent tmux panes")
@click.option("--smoke", is_flag=True, help="Pass --smoke to every child run (fast_dev_run)")
@click.option("--ablation", "-a", is_flag=True, help="Run ablation studies for each experiment")
@click.option("--experiments", "-e", multiple=True, help="Specific experiment names to run")
@click.option("--exclude", "-ex", multiple=True, help="Experiment names to exclude")
@click.option("--gpu", "-g", type=int, default=0, help="Force a single GPU index")
def cli(config: Path, batch: int, smoke: bool, ablation: bool, experiments: List[str], 
        exclude: List[str], gpu: int):
    
    if not shutil.which("tmux"):
        console.print("[red]❌ tmux not found.")
        sys.exit(1)

    # Load the experiment configuration
    cfg = OmegaConf.load(config)
    
    # Get experiment list
    if experiments:
        # Use specific experiments provided by user
        experiment_list = list(experiments)
    elif 'experiment_list' in cfg:
        # Use the list from the config file
        experiment_list = cfg.experiment_list
    else:
        # Use all experiments defined in the config
        experiment_list = list(cfg.experiments.keys())
    
    # Apply exclusions
    experiment_list = [e for e in experiment_list if e not in exclude]
    
    if not experiment_list:
        console.print("[red]❌ No experiments to run after exclusions")
        sys.exit(1)

    smoke_flag = "--smoke" if smoke else ""

    # Process experiments
    experiment_names = []
    config_paths = []
    parameters = []
    
    for exp_name in experiment_list:
        if exp_name not in cfg.experiments:
            console.print(f"[yellow]⚠ Experiment '{exp_name}' not found in config, skipping")
            continue
            
        # Get the experiment config
        exp_config = cfg.experiments[exp_name]
        
        # Apply experiment-specific overrides to defaults
        merged_config = OmegaConf.merge(OmegaConf.create(cfg.defaults), OmegaConf.create(exp_config))
        
        # If ablation flag is set, generate an experiment for each ablation
        if ablation:
            # Find all ablation configs
            ablation_dir = Path("histopathology/configs/ablations")
            if not ablation_dir.exists():
                console.print(f"[yellow]⚠ Ablation directory {ablation_dir} not found")
                continue
                
            ablation_files = list(ablation_dir.glob("*.yaml"))
            
            for abl_file in ablation_files:
                abl_name = abl_file.stem
                
                # Create a unique name for this experiment+ablation
                exp_abl_name = f"{exp_name}_{abl_name}"
                
                # Load ablation config
                abl_cfg = OmegaConf.load(abl_file)
                
                # Merge experiment with ablation
                final_cfg = OmegaConf.merge(merged_config, abl_cfg)
                
                # Ensure WandB tags reflect both experiment and ablation
                if 'wandb' in final_cfg and 'tags' in final_cfg.wandb:
                    final_cfg.wandb.tags.append(exp_name)
                    final_cfg.wandb.name = exp_abl_name
                
                # Write temporary config file
                with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
                    OmegaConf.save(final_cfg, tf.name)
                    exp_config_path = Path(tf.name)
                
                # Add to experiment list
                experiment_names.append(exp_abl_name)
                config_paths.append(exp_config_path)
                parameters.append(_param_size(final_cfg))
        else:
            # Standard experiment without ablations
            # Write temporary config file
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
                OmegaConf.save(merged_config, tf.name)
                exp_config_path = Path(tf.name)
            
            # Add to experiment list
            experiment_names.append(exp_name)
            config_paths.append(exp_config_path)
            parameters.append(_param_size(merged_config))
    
    if not experiment_names:
        console.print("[red]❌ No valid experiments after processing")
        sys.exit(1)

    # Create unique session name
    session = f"{SESSION_BASE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _tmux("kill-session", "-t", session)
    _tmux("new-session", "-d", "-s", session, "-n", "dashboard", "watch -n1 nvidia-smi")
    _tmux("set-option", "-g", "remain-on-exit", "off")
    
    exp_type = "ablation experiments" if ablation else "experiments"
    console.rule(f"🚀 Launching {len(experiment_names)} {exp_type}")
    console.print(f"Experiments: [cyan]{', '.join(experiment_names)}[/]")

    try:
        _scheduler(experiment_names, config_paths, parameters, batch, gpu, smoke_flag, session)
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted — tearing down tmux session...")
        _tmux("kill-session", "-t", session)
        sys.exit(130)
    finally:
        # Clean up temporary files
        for config_path in config_paths:
            try:
                if config_path.exists():
                    config_path.unlink()
            except:
                pass

if __name__ == "__main__":
    cli()
