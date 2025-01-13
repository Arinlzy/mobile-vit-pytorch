import os 

datasets = ['cifar10', 'stl10']
width_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
device = 2
log_dir = f"log/width_multiplier"

for wm in width_multipliers:
    for dataset in datasets:
            os.system(f"python main.py --dataset {dataset} --width_multiplier {wm} --device {device} --log_path {log_dir}")
