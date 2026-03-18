import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
import datetime

# 导入模块
from src.clip.dataset import MMClipDataset
from src.clip.model import SimpleMMClip, contrastive_loss 

def main():
    # 1. 配置
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEXT_MODEL = "bert-base-uncased" # 或者 "sentence-transformers/all-MiniLM-L6-v2"
    
    #日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"clip_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    print(f"Logging to {log_dir}")

    # 2. 数据准备
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = MMClipDataset(
        json_file='/root/jyz/my_mmLLM/processed_dataset/train.json',
        root_dir='/root/jyz/my_mmLLM/processed_dataset/',
        tokenizer=tokenizer,
        transform=transform
    )
    
    # 同样记得 num_workers=0 或 2~4 根据情况调整
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    
    # 3. 模型初始化
    model = SimpleMMClip(text_model_name=TEXT_MODEL).to(DEVICE)
    
    # === 新增策略：前 5 个 Epoch 冻结 Backbone ===
    print("🥶 Pattern: Freezing Backbones for the first 5 epochs...")
    for param in model.radar_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
    # 只优化投影层和 temperature
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-3) # 投影层可以用大一点的 LR
    
    scaler = torch.cuda.amp.GradScaler()

    # 4. 训练循环
    print("Start Training CLIP...")
    for epoch in range(EPOCHS):
        
        # === 第 5 个 Epoch 解冻 ===
        if epoch == 5:
            print("🔥 Unfreezing Backbones...")
            for param in model.parameters():
                param.requires_grad = True
            # 解冻后使用较小的 LR 微调
            optimizer = optim.AdamW(model.parameters(), lr=1e-5) 

        model.train()
        total_loss = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(loop): # 修改这里获取 batch_idx
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # === 诊断代码 ===
            if batch_idx == 0 and epoch == 0:
                img_std = pixel_values.std()
                print(f"\n[DIAGNOSTIC] Batch 0 Stats:")
                print(f"  Img Shape: {pixel_values.shape}")
                print(f"  Img Mean: {pixel_values.mean():.4f}, Max: {pixel_values.max():.4f}")
                print(f"  Img Std: {img_std:.4f}")
                if img_std < 1e-6:
                    raise RuntimeError("❌ 严重错误：输入数据标准差为0（可能是全黑图或路径错误），模型无法学习！请检查 dataset.py 中的路径逻辑。")
            # =================
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                radar_emb, text_emb, scale = model(pixel_values, input_ids, attention_mask)
                loss = contrastive_loss(radar_emb, text_emb, scale)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, "last.pth"))
        
        # 可选：保存 Radar Encoder 单独权重 (Stage 2 需要这个)
        torch.save(model.radar_encoder.state_dict(), os.path.join(log_dir, "radar_encoder_only.pth"))

    print("Training Finished.")

if __name__ == "__main__":
    main()