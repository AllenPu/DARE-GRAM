#!/usr/bin/env bash
python train_source.py --gpu_id 0 --src n --tgt s --tradeoff 0.001 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src n --tgt c --tradeoff 0.001 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src s --tgt n --tradeoff 0.001 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src s --tgt c --tradeoff 0.003 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src c --tgt s --tradeoff 0.003 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src c --tgt n --tradeoff 0.003 --tradeoff2 0.03 