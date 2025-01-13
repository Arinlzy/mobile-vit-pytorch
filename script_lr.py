import os

datasets = ['cifar10', 'stl10']
lrs = [0.001, 0.003, 0.01, 0.03, 0.1]
device = 1
log_dir = f"log/lr"

for lr in lrs:
    for dataset in datasets:
            os.system(f"python main.py --dataset {dataset} --lr {lr} --device {device} --log_path {log_dir}")

