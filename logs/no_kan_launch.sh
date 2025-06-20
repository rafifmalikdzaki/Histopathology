#!/bin/bash
source logs/no_kan_env.sh
cd /home/dzakirm/Research/Histopathology
python -m histopathology.src.training.dae_kan_attention.pl_training_robust --config histopathology/configs/ablations/no_kan.yaml --smoke 2>&1 | tee logs/no_kan.log
