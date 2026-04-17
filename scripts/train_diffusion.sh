#!/bin/bash
accelerate launch --config_file configs/accelerate.yaml train_diffusion.py --config configs/train_diffusion.yaml