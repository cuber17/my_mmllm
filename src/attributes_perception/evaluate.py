import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MMWaveAttributeDataset # type: ignore
from model import MultiHeadAttributeClassifier # type: ignore
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _get_class_names_for_task(task_name, task_map, num_classes):
    if isinstance(task_map, dict):
        # 优先使用 id->label 的稳定顺序
        if len(task_map) > 0 and all(isinstance(v, int) for v in task_map.values()):
            sorted_items = sorted(task_map.items(), key=lambda x: x[1])
            names = [name for name, _ in sorted_items]
        else:
            names = [str(k) for k in task_map.keys()]
    elif isinstance(task_map, list):
        names = [str(x) for x in task_map]
    else:
        names = []

    if len(names) != num_classes:
        names = [f"Class_{i}" for i in range(num_classes)]

    return names


def _save_confusion_matrix_image(cm, class_names, task_name, save_path):
    cm_np = cm.cpu().numpy()
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_np,
        row_sums,
        out=np.zeros_like(cm_np, dtype=np.float64),
        where=row_sums != 0,
    )

    num_classes = len(class_names)
    fig_size = min(max(8, num_classes * 0.7), 24)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)

    ax.set_title(f"Confusion Matrix - {task_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # 类别太多时只画热力图，避免文字重叠
    if num_classes <= 15:
        threshold = cm_norm.max() / 2.0 if cm_norm.size > 0 else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    f"{cm_np[i, j]}\n{cm_norm[i, j] * 100:.1f}%",
                    ha='center',
                    va='center',
                    color='white' if cm_norm[i, j] > threshold else 'black',
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Row-normalized ratio')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def evaluate():
    # --- 1. 配置 ---
    EXPERIMENT_ID = "attributes_20260405_052021" # 请确认这是你要评测的权重文件夹
    PROJECT_ROOT = '/root/jyz/my_mmLLM'
    
    # [FIX] 数据集的根目录应该是 processed_dataset
    # 这样拼接 json 里的 "./imgs_test/" 才能找到正确文件
    DATASET_ROOT = os.path.join(PROJECT_ROOT, 'processed_dataset')

    # 路径配置
    BASE_LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    EXP_DIR = os.path.join(BASE_LOG_DIR, EXPERIMENT_ID)
    
    CHECKPOINT_PATH = os.path.join(EXP_DIR, "best.pth")
    LABEL_MAP_PATH = os.path.join(EXP_DIR, "label_maps.json")
    
    JSON_FILE = os.path.join(DATASET_ROOT, 'test.json')
    
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading experiment from: {EXP_DIR}")
    print(f"Dataset root: {DATASET_ROOT}")  # 打印一下供检查
    
    # 2. 加载 Label Maps
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}")
        
    with open(LABEL_MAP_PATH, 'r') as f:
        label_maps = json.load(f)
    
    output_dims = {k: len(v) for k, v in label_maps.items()}
    
    # 3. 初始化模型并加载权重
    model = MultiHeadAttributeClassifier(output_dims).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. 数据加载
    # 评估时只做 Resize + Normalize
    # 注意: 这里去掉了 Resize，因为 Dataset 内部已经有了 Resize 逻辑 (参考 dataset.py)
    # 如果 dataset.py 没有 resize，这里需要加 transforms.Resize((224, 224), antialias=True)
    # 既然你之前的 dataset.py 代码里有 Resize，这里就只留 Normalize
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    
    # [FIX] 这里传入 DATASET_ROOT
    dataset = MMWaveAttributeDataset(JSON_FILE, DATASET_ROOT, transform=transform, label_maps=label_maps)
    
    # Num_workers=1 便于调试，如果路径还有问题会直接报错而不是挂起
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    # 5. 开始评估
    correct_counts = {k: 0 for k in output_dims.keys()}
    confusion_matrices = {
        k: torch.zeros((output_dims[k], output_dims[k]), dtype=torch.int64)
        for k in output_dims.keys()
    }
    total_samples = 0
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    with torch.no_grad():
        for imgs, labels_dict in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            target_labels = {k: v.to(DEVICE) for k, v in labels_dict.items()}
            
            preds = model(imgs)
            total_samples += imgs.size(0)
            
            for k in preds.keys():
                _, predicted = torch.max(preds[k], 1)
                targets = target_labels[k]
                correct_counts[k] += (predicted == targets).sum().item()

                # 将 (true, pred) 对映射到混淆矩阵计数
                num_classes = output_dims[k]
                encoded = (targets * num_classes + predicted).detach().cpu()
                batch_cm = torch.bincount(encoded, minlength=num_classes * num_classes)
                confusion_matrices[k] += batch_cm.reshape(num_classes, num_classes)
    
    # 6. 打印结果
    print("\n" + "="*40)
    print(f"Experiment: {EXPERIMENT_ID}")
    print("Evaluation Results (Top-1 Accuracy)")
    print("="*40)
    
    avg_acc = 0
    for k in output_dims.keys():
        acc = correct_counts[k] / total_samples
        print(f"{k:<20}: {acc*100:.2f}%")
        avg_acc += acc
        
    # 计算平均准确率
    if len(output_dims) > 0:
        print("-" * 40)
        print(f"Average Accuracy    : {(avg_acc/len(output_dims))*100:.2f}%")
    print("="*40)

    # 7. 导出混淆矩阵图片
    cm_dir = os.path.join(EXP_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    print("\nSaving confusion matrix images...")
    for task_name, cm in confusion_matrices.items():
        class_names = _get_class_names_for_task(
            task_name,
            label_maps.get(task_name, {}),
            output_dims[task_name],
        )
        out_path = os.path.join(cm_dir, f"{task_name}_confusion_matrix.png")
        _save_confusion_matrix_image(cm, class_names, task_name, out_path)
        print(f"- {task_name}: {out_path}")

if __name__ == "__main__":
    evaluate()