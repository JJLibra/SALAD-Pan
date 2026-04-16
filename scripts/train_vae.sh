#!/bin/bash
accelerate launch --config_file config/accelerate.yaml train_vae.py --config config/train_vae.yaml