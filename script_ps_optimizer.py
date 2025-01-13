import os

datasets = ['cifar10', 'stl10']
optimizers = ['adam', 'sgd']
patch_sizes = [1, 2, 3]
device = 3

for ps in patch_sizes:
    dataset = 'stl10'
    log_dir = f"log/patch_size"
    os.system(f"python main.py --dataset {dataset} --patch_size {ps} {ps} --device {device} --log_path {log_dir}")

for optimizer in optimizers:
    for dataset in datasets:
        log_dir = f"log/optimizer"
        os.system(f"python main.py --dataset {dataset} --optimizer {optimizer} --device {device} --log_path {log_dir}")