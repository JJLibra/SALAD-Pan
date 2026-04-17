#!/bin/bash
accelerate launch --config_file configs/accelerate.yaml train_vae.py --config configs/train_vae.yaml