#load pytorch
module load pytorch/1.11.0

#Default command to run
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch perceiver --output_dir output_04162023_trial2 --epochs=1000

#Evaluating DINO with eval_linear.py
#python eval_linear.py --arch perceiver --pretrained_weights output_04152023_trial5/checkpoint0080.pth
#python eval_linear.py --arch perceiver --output_dir output_04152023_trial3 --pretrained_weights output_04152023_trial3/checkpoint0009.pth

#Evaluating DINO with eval_knn.py
#python -m torch.distributed.launch --nproc_per_node=4 eval_knn.py --arch vit_small --pretrained_weights output_04152023_trial5/checkpoint0080.pth

#python eval_knn.py --arch perceiver --pretrained_weights output_04152023_trial5/checkpoint0080.pth
#Visualizing attention mask with visualize_attention.py
#python visualize_attention.py --pretrained_weights output_04152023_trial5/checkpoint0080.pth --output_dir output_04152023_trial5/