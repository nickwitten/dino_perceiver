#load pytorch
module load pytorch/1.11.0

#Default command to run
python -m torch.distributed.launch --nproc_per_node=4 main_dino_cifar100.py --arch perceiver --output_dir output_04162023_trial3_cifar100 --epochs=1000
