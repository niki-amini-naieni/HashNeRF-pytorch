#!/bin/sh
for run in {1..8}; do
  CUDA_VISIBLE_DEVICES=0 python run_nerf.py --expname seed_0 --data_split_file "next_best_view_llff_data_splits.json" --seed 0 --iters 2000 &
  CUDA_VISIBLE_DEVICES=1 python run_nerf.py --expname seed_1 --data_split_file "next_best_view_llff_data_splits.json" --seed 1 --iters 2000 &
  CUDA_VISIBLE_DEVICES=2 python run_nerf.py --expname seed_2 --data_split_file "next_best_view_llff_data_splits.json" --seed 2 --iters 2000 &
  CUDA_VISIBLE_DEVICES=3 python run_nerf.py --expname seed_3 --data_split_file "next_best_view_llff_data_splits.json" --seed 3 --iters 2000 &
  CUDA_VISIBLE_DEVICES=4 python run_nerf.py --expname seed_4 --data_split_file "next_best_view_llff_data_splits.json" --seed 4 --iters 2000 
  python next_best_view_eval.py
done
