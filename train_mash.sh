torchrun \
	--nnodes=1 \
	--nproc_per_node=7 \
	--rdzv_id=100 \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$127.0.0.1:29400 \
	train_mash.py
