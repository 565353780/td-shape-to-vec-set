cd ..
git clone git@github.com:565353780/base-trainer.git

cd base-trainer
./dev_setup.sh

pip install -U h5py timm einops transformers scipy trimesh

pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
