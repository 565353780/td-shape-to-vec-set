export CUDA_VISIBLE_DEVICES=0 && OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 train_mash.py
