import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.attributes_perception.dataset import MMWaveAttributeDataset
from src.attributes_perception.model import MultiHeadAttributeClassifier
import os
import json
import logging
import datetime
from pathlib import Path
from tqdm import tqdm

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_dataset_root(project_root: Path) -> Path:
    candidates = [
        project_root / 'processed_dataset_rebalanced',
        project_root / 'processed_dataset',
    ]

    for root in candidates:
        train_json = root / 'train.json'
        has_images = (root / 'imgs').exists() or (root / 'imgs_test').exists()
        if train_json.exists() and has_images:
            return root

    raise FileNotFoundError(
        "No valid dataset root found. Need train.json and imgs/imgs_test under processed_dataset_rebalanced or processed_dataset."
    )

def train():
    # --- 1. 配置与环境 ---
    # 定义实验目录
    TIMESTAMP = get_timestamp()
    BASE_LOG_DIR = 'logs'
    LOG_DIR = os.path.join(BASE_LOG_DIR, f'attributes_{TIMESTAMP}')
    os.makedirs(LOG_DIR, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    project_root = Path('/root/jyz/my_mmLLM')
    ROOT_DIR = str(resolve_dataset_root(project_root))
    JSON_FILE = os.path.join(ROOT_DIR, 'train.json')
    
    # 超参数
    BATCH_SIZE = 64  # AMP 开启后显存占用降低，可以尝试加大 Batch Size (原32)
    LR = 1e-4
    EPOCHS = 20
    # 注意：如果再次遇到 Bus Error，请将 num_workers 改回 0 或者 1
    # 推荐: 4, 并开启 persistent_workers
    NUM_WORKERS = 1 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Start training on {DEVICE}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Dataset root: {ROOT_DIR}")
    
    # --- 2. 数据准备 ---
    # 定义 Transform: 只做 Normalize，因为 Dataset 内部已经做过 MinMax 和 Resize(224)
    # 将 [0, 1] 映射到 [-1, 1] 有助于神经网络收敛
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    label_maps_path = os.path.join(ROOT_DIR, 'label_maps.json')
    if os.path.exists(label_maps_path):
        with open(label_maps_path, 'r') as f:
            label_maps = json.load(f)
    else:
        label_maps = None

    dataset = MMWaveAttributeDataset(JSON_FILE, ROOT_DIR, transform=transform, label_maps=label_maps)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,        # 加速 CPU -> GPU 传输
        persistent_workers=(NUM_WORKERS > 0) # 避免每个 Epoch 重建 Worker
    )
    
    # 保存 Label Map (只需保存一次)
    with open(os.path.join(LOG_DIR, 'label_maps.json'), 'w') as f:
        json.dump(dataset.label_maps, f, indent=4)
        
    output_dims = {k: len(v) for k, v in dataset.label_maps.items()}
    logger.info(f"Classes per attribute: {output_dims}")
    
    # --- 3. 模型、优化器与 AMP ---
    model = MultiHeadAttributeClassifier(output_dims).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # 初始化 GradScaler 用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    
    # --- 4. 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        # 使用 tqdm 显示进度条，但不频繁打印详细 dict
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        
        for batch_idx, (imgs, labels_dict) in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            target_labels = {k: v.to(DEVICE, non_blocking=True) for k, v in labels_dict.items()}
            
            optimizer.zero_grad()
            
            # 开启混合精度上下文
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                
                loss = torch.tensor(0.0, device=DEVICE)
                loss_detail = {}
                for k in preds.keys():
                    l = criterion(preds[k], target_labels[k])
                    loss += l
                    # items() 会导致 GPU 同步，仅在需要打印时读取
                    # loss_detail[k] = l.item() 
            
            # 使用 scaler 进行反向传播和步进
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            curr_loss = loss.item()
            total_epoch_loss += curr_loss
            
            # 减少打印频率：进度条只显示总 Loss
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=curr_loss)
            
            # 每 100 step 记录一次详细日志到文件（不在控制台刷屏）
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch} Step {batch_idx}: Total Loss {curr_loss:.4f}")

        avg_loss = total_epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} Completed. Average Loss: {avg_loss:.4f}")
        
        # 保存策略：
        # 1. 保存最新的 (last)
        torch.save(model.state_dict(), os.path.join(LOG_DIR, "last.pth"))
        
        # 2. 保存最好的 (best)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "best.pth"))
            logger.info(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == '__main__':
    train()