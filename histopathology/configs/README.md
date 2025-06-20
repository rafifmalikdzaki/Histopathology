# Histopathology DAE-KAN-Attention Configuration System

This directory contains configuration files for training and evaluating the Histopathology DAE-KAN-Attention models. The configuration system is designed to support multiple experiments, ablation studies, and parameter variations.

## Configuration Structure

The configuration system is organized hierarchically:

```
configs/
├── wandb_config.yaml          # Main WandB configuration
├── experiments.yaml           # Multiple experiment definitions
├── README.md                  # This documentation
└── ablations/                 # Ablation study configurations
    ├── base.yaml              # Base ablation configuration
    ├── no_kan.yaml            # Model without KAN layers
    ├── no_bam.yaml            # Model without BAM attention
    └── no_eca.yaml            # Model without ECA attention
```

## Experiments Configuration (experiments.yaml)

The `experiments.yaml` file defines multiple experiment variations that can be run sequentially or in parallel. It contains:

1. **Hardware settings** - GPU and worker configurations
2. **Default settings** - Base configuration for all experiments
3. **Experiment variations** - Different hyperparameter combinations to test
4. **Experiment list** - Which experiments to run by default

### Key Sections

- `defaults`: Common settings across all experiments
- `experiments`: Individual experiment definitions with variations
- `experiment_list`: List of experiments to run with the launcher

### Adding a New Experiment

To add a new experiment, add a new entry to the `experiments` section:

```yaml
experiments:
  # Existing experiments...
  
  # Your new experiment
  my_new_experiment:
    name: "my_new_experiment"
    description: "Description of what you're testing"
    # Override only the parameters you want to change
    optimizer:
      lr: 3.0e-4
    training:
      batch_size: 24
```

Then add it to the `experiment_list` to include it in the default run set:

```yaml
experiment_list:
  # Existing experiments...
  - my_new_experiment
```

## Ablation Configurations

Ablation studies allow testing the importance of specific model components. Files in the `ablations/` directory define different model variations:

- `base.yaml`: The baseline configuration with all components enabled
- `no_kan.yaml`: Removes KAN layers to test their impact
- `no_bam.yaml`: Removes Bottleneck Attention Module
- `no_eca.yaml`: Removes Efficient Channel Attention

Each ablation config inherits from `base.yaml` and overrides specific parameters.

### Creating a New Ablation Study

To create a new ablation configuration:

1. Create a new YAML file in the `ablations/` directory
2. Inherit from the base configuration
3. Override specific model parameters
4. Add identification and tracking information

Example:

```yaml
# ablations/my_new_ablation.yaml
defaults:
  - base
  - _self_

# Ablation identifier
ablation:
  id: "my_new_ablation"
  description: "Description of this ablation"

# Override model parameters
model:
  use_my_feature: false

# Ensure consistent settings
wandb:
  tags: ['autoencoder', 'kan', 'histopathology', 'ablation', 'my_new_ablation']
  log_artifacts: false

callbacks:
  model_checkpoint:
    save_top_k: 1
```

## Running Experiments with the Launcher

The project includes two launcher scripts for running experiments:

1. `launch_tmux_ablation.py`: Focused on ablation studies
2. `launch_tmux_experiments.py`: For running multiple experiments with optional ablations

### Experiment Launcher Usage

The experiment launcher supports various options:

```bash
# Run all experiments defined in experiment_list
python launch_tmux_experiments.py

# Run specific experiments
python launch_tmux_experiments.py -e baseline high_lr low_lr

# Run with ablation studies for each experiment
python launch_tmux_experiments.py -a

# Fast dev run (smoke test)
python launch_tmux_experiments.py --smoke -e baseline

# Control GPU and batch size
python launch_tmux_experiments.py -g 1 -b 3
```

### Ablation Launcher Usage

For focused ablation studies:

```bash
# Run all ablations
python launch_tmux_ablation.py

# Run with custom base config
python launch_tmux_ablation.py -c histopathology/configs/custom_config.yaml

# Exclude specific ablations
python launch_tmux_ablation.py -ex no_kan
```

## Common Use Cases and Examples

### Training the Base Model

```bash
python launch_tmux_experiments.py -e baseline
```

### Running a Learning Rate Sweep

```bash
python launch_tmux_experiments.py -e high_lr baseline low_lr
```

### Testing Different Noise Levels

```bash
python launch_tmux_experiments.py -e high_noise baseline low_noise
```

### Comparing All Ablations on the Baseline Model

```bash
python launch_tmux_ablation.py
```

### Running Ablation Studies for Multiple Experiments

```bash
python launch_tmux_experiments.py -e baseline high_lr -a
```

This will run both the baseline and high learning rate experiments for each ablation configuration.

### Debugging with Smoke Tests

```bash
python launch_tmux_experiments.py --smoke -e baseline
```

This runs a fast dev run with reduced epochs to quickly check for errors.

## Important Notes

1. The `save_top_k` parameter is hard-coded to 1 in the training script
2. WandB artifact logging is disabled by default
3. Experiment results are tracked in real-time in tmux sessions
4. Log files are saved in the `logs/` directory
5. Model checkpoints are saved according to the best validation PSNR

## Common Issues and Solutions

- **Issue**: tmux session not starting
  - **Solution**: Make sure tmux is installed (`sudo apt install tmux` or equivalent)

- **Issue**: CUDA out of memory
  - **Solution**: Reduce batch size or use smaller models (`-e small_batch`)

- **Issue**: Can't see real-time progress
  - **Solution**: Attach to the tmux session with `tmux attach -t <session_name>`
