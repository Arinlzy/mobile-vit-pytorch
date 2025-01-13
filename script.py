import os
import multiprocessing
import itertools

def run_experiment(args):
    gpu_id, dataset, lr, optimizer, width_multiplier, patch_size = args    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置 GPU
    
    log_dir = f"logs/{dataset}_lr{lr}_opt{optimizer}_wm{width_multiplier}_ps{patch_size}"
    os.makedirs(log_dir, exist_ok=True)

    command = f"python main.py --dataset {dataset} --lr {lr} --optimizer {optimizer} --width_multiplier {width_multiplier} --patch_size {patch_size} --log_path {log_dir} --device {gpu_id}"
    os.system(command)


if __name__ == "__main__":
    # 参数配置
    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # 可用的 GPU ID 列表
    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]
    optimizers = ["adam", "sgd"]
    width_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    datasets = ["cifar10", "stl10"]
    patch_sizes = [(1, 1), (2, 2), (3, 3)]


    # 生成参数组合, 用 itertools.product 生成所有参数组合
    param_combinations_cifar = list(itertools.product('cifar10', learning_rates, optimizers, width_multipliers, (1, 1)))



#     # 为每个任务分配 GPU
#     tasks = []
#     for i, params in enumerate(param_combinations):
#         gpu_id = available_gpus[i % len(available_gpus)]  # 循环分配 GPU
#         tasks.append((gpu_id, *params))
# 
#     # 使用 multiprocessing 并行运行
#     with multiprocessing.Pool(processes=len(available_gpus)) as pool:  # 每个 GPU 一个进程
#         pool.map(run_experiment, tasks)
# 
#     print("All experiments finished.")
    print(param_combinations)
