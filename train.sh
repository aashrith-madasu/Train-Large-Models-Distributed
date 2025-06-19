#!/bin/bash

accelerate launch --config_file configs/fsdp_mp_config.yaml train.py