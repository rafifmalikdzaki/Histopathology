#!/bin/bash
source logs/base_env.sh
cd /home/dzakirm/Research/Histopathology
python -m histopathology.src.training.dae_kan_attention.pl_training_robust --config histopathology/configs/ablations/base.yaml --smoke 2>&1 | tee logs/base.log
