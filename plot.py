import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 日志文件路径
log_dir = "optimizer_stl10"  # 修改为你的日志文件目录
log_pattern = re.compile(r"Epoch (\d+) - Eval Loss: ([\d.]+), Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)")

def extract_metrics_from_logs(log_dir):
    data = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                log_file_path = os.path.join(root, file)
                # 从文件名中提取 optimizer
                optimizer_match = re.search(r"_(adam|sgd)_", file)
                if not optimizer_match:
                    continue
                optimizer = optimizer_match.group(1)
                with open(log_file_path, 'r') as f:
                    for line in f:
                        match = log_pattern.search(line)
                        if match:
                            epoch, loss, accuracy, precision, recall, f1 = match.groups()
                            data.append({
                                "Log File": file,
                                "Optimizer": optimizer,
                                "Epoch": int(epoch),
                                "Loss": float(loss),
                                "Accuracy": float(accuracy),
                                "Precision": float(precision),
                                "Recall": float(recall),
                                "F1-Score": float(f1)
                            })
    return pd.DataFrame(data)

def plot_metrics_comparison(metrics_df):
    metrics = ["Loss", "Accuracy", "Precision", "Recall", "F1-Score"]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for optimizer in metrics_df["Optimizer"].unique():
            optimizer_data = metrics_df[metrics_df["Optimizer"] == optimizer]
            ax.plot(optimizer_data["Epoch"], optimizer_data[metric], label=f"Optimizer={optimizer}", marker='o')

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    # 设置整体标题
    fig.suptitle("Comparison of Evaluation Metrics Across Optimizers On STL10", fontsize=16)

    # 保存图像
    plt.savefig("comparison_metrics_optimizer.png")
    plt.show()

# 执行流程
metrics_df = extract_metrics_from_logs(log_dir)
if not metrics_df.empty:
    plot_metrics_comparison(metrics_df)
else:
    print("No metrics found in log files.")
