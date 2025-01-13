import argparse
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from mobile_vit import mobilevit_v3_v2


# 设置日志记录

def setup_logging(args):
    log_filename = f"{args.dataset}_lr{args.lr}_{args.optimizer}_wm{args.width_multiplier}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    # 如果log文件夹不存在，则创建
    log_path = os.path.join(args.log_path, log_filename)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # 如果log文件已经存在，则删除
    if os.path.exists(log_path):
        os.remove(log_path)
        
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(f"Log file created at {log_path}")
    return logger


# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileViT on CIFAR-10")

    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--log_path', type=str, default='log', help='Path to save log files')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to train on')

    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--width_multiplier', type=float, default=1, help='Width multiplier for MobileViT',
                        choices= [0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    parser.add_argument('--patch_size', type=int, nargs=2, default=(1, 1), help='Patch size for MobileViT')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and validation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training progress')

    return parser.parse_args()

def get_data(args):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        return train_loader, test_loader
    
    elif args.dataset == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
        ])

        train_dataset = torchvision.datasets.STL10(
            root='./data', split='train', download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.STL10(
            root='./data', split='test', download=True, transform=transform_test
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        return train_loader, test_loader

    else:
        raise ValueError("Dataset not supported!")


def get_model(args):
    if args.dataset == 'cifar10':
        model = mobilevit_v3_v2.MobileViTv3_v2(
            image_size=(32, 32),
            width_multiplier=args.width_multiplier,
            num_classes=10,
            patch_size=(1, 1) # cifar10上只能使用(1,1)
        )
        return model
    
    elif args.dataset == 'stl10':
        model = mobilevit_v3_v2.MobileViTv3_v2(
            image_size=(96, 96),
            width_multiplier=args.width_multiplier,
            num_classes=10,
            patch_size=args.patch_size
        )
        return model
    else:
        raise ValueError("Dataset not supported!")

def get_optimizer(model, args):
    if args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError("Optimizer not supported!")

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, logger):
    model.train()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录真实值和预测值
        _, predicted = outputs.max(1)
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # if batch_idx % args.log_interval == 0:
        #     logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, Loss: {total_loss / (batch_idx + 1):.3f}")

    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions, average='macro')
    precision = precision_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    logger.info(f"Epoch {epoch} - Train Loss: {total_loss / len(data_loader):.4f}, " +
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return total_loss / len(data_loader), precision, recall, f1


def evaluate(model, criterion, data_loader, device, logger, epoch):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            # 记录真实值和预测值
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions, average='macro')
    precision = precision_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    logger.info(f"Epoch {epoch} - Eval Loss: {total_loss / len(data_loader):.4f}, " +
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return total_loss / len(data_loader), precision, recall, f1

def main():
    args = parse_args()
    logger = setup_logging(args)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    train_loader, test_loader = get_data(args)
    model = get_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args, logger)
        evaluate(model, criterion, test_loader, device, logger, epoch)
        
    logger.info("Training finished")


if __name__ == '__main__':
    main()
