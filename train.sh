#!/bin/bash

accelerate launch --config_file configs/ddp_mp_config.yaml train.py