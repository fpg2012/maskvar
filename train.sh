#!/bin/bash

torchrun --nproc_per_node=1 --nnodes=4 --master_addr=127.0.0.1 --master_port=11134 maskvar_train.py
