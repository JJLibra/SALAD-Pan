#!/bin/bash
accelerate launch --config_file config/accelerate.yaml train_diffusion.py --config config/train_diffusion.yaml