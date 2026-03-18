import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MMWaveAttributeDataset # type: ignore
from model import MultiHeadAttributeClassifier # type: ignore
import os
import json
from tqdm import tqdm

def evaluate():
    # --- 1. 配置 ---
    EXPERIMENT_ID = "attributes_20260131_145653" # 请确认这是你要评测的权重文件夹
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
                correct_counts[k] += (predicted == target_labels[k]).sum().item()
    
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

if __name__ == "__main__":
    evaluate()