#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=11134 maskvar_train.py --exp_name=flex_train1 --local_out_dir ./test-maskvar_flex-1 --bs=4 --ep=4 --fp16=2 --vfast=1 --tfast=1