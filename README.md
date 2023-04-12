# dino_perceiver
To run the dino algorithm with the perciever model, navigate to the dino folder and run the main_dino.py script with the following configuration:

`python main_dino.py --arch perceiver --output_dir ..\dino-sample-output\ --batch_size_per_gpu=1 --epochs=10 --num_workers=1 `


Setup for Running with Pace ICE GPUs
- module load pytorch/1.11.0
- pip install perciever-pytorch
- pip install timm

Use the command below to ceate an interactive session (set number of GPUs)
`pace-interact -q coc-ice-gpu -l nodes=1:ppn=12:gpus=1 `