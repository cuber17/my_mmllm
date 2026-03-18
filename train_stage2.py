import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import timm
from tqdm import tqdm
import os
import datetime
import logging

# 引用自定义模块
from src.llm.projector import RadarProjector
from src.llm.dataset import MMLLMDataset

def main():
    # --- 1. 核心配置 ---
    # 路径配置
    RADAR_ENCODER_PATH = "logs/clip_20260120_224659/radar_encoder_only.pth"  # 修改为你的 Stage 1 路径
    LLM_MODEL_PATH = "./huggingface/Phi-3-mini-4k-instruct"
    
    # 训练配置
    BATCH_SIZE = 4       # 根据显存调整，3090/4090 可以设 4-8
    GRAD_ACCUM = 4       # 梯度累积，等效 Batch Size = 16
    LR_PROJ = 1e-3       # Projector 学习率大一点
    LR_LLM = 2e-4        # LoRA 学习率
    EPOCHS = 10
    DEVICE = "cuda"

    # 日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"stage2_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    print(f"Logging to {log_dir}")

    # --- 2. 模型加载 ---
    print(">>> Loading Radar Encoder (Frozen)...")
    radar_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    # 加载权重 (处理可能的 key 前缀不匹配)
    ckpt = torch.load(RADAR_ENCODER_PATH)
    # 如果 key 有 "radar_encoder." 前缀，去除它
    new_ckpt = {k.replace("radar_encoder.", ""): v for k, v in ckpt.items()}
    msg = radar_encoder.load_state_dict(new_ckpt, strict=False)
    print(f"Encoder Load Msg: {msg}")
    
    radar_encoder.to(DEVICE).eval()
    for param in radar_encoder.parameters():
        param.requires_grad = False # 冻结

    print(">>> Loading LLM (Phi-3) & LoRA...")
    # 修改 1: trust_remote_code=False (使用 transformers 库内置的稳定实现，不使用模型文件夹里的旧代码)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=False)
    # Phi-3 没有默认的 pad_token
    tokenizer.pad_token = tokenizer.eos_token 
    
    # 修改 2: trust_remote_code=False, 并将 torch_dtype 改为 dtype (修复警告)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH, 
        trust_remote_code=False, 
        dtype=torch.bfloat16,   # 从 torch_dtype 改为 dtype
        device_map="auto"
    )
    
    # 应用 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)
    llm.print_trainable_parameters()

    print(">>> Initializing Projector...")
    # ViT-Base=768, Phi-3=3072
    projector = RadarProjector(encoder_dim=768, llm_dim=3072).to(DEVICE)

    # --- 3. 数据集 ---
    dataset = MMLLMDataset(
        json_file='/root/jyz/my_mmLLM/processed_dataset/train.json',
        root_dir='/root/jyz/my_mmLLM/processed_dataset/'
    )
    
    # Pad Collate Function
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        conversations = [item['conversation'] for item in batch]
        return pixel_values, conversations

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # --- 4. 优化器 ---
    optimizer = torch.optim.AdamW([
        {'params': projector.parameters(), 'lr': LR_PROJ},
        {'params': llm.parameters(), 'lr': LR_LLM}
    ])

    # --- 5. 训练循环 ---
    print(">>> Start Training...")
    projector.train()
    llm.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, (pixel_values, conversations) in enumerate(loop):
            pixel_values = pixel_values.to(DEVICE, dtype=torch.bfloat16)  # 转为 bfloat16 匹配 LLM
            
            # --- Forward Pass ---
            # 1. 获取 Radar Embeddings
            with torch.no_grad():
                # [B, 197, 768] (包含 CLS token)
                # Encoder 是 float32 的，所以输入要转 float()
                radar_feats = radar_encoder.forward_features(pixel_values.float()) 
            
            # 2. Project 到 LLM 维度 -> [B, 197, 3072]
            # Projector 目前是 float32，输出也是 float32
            radar_embeds = projector(radar_feats.to(DEVICE)) 
            
            # === 核心修复: 转换 dtype ===
            radar_embeds = radar_embeds.to(dtype=torch.bfloat16)
            
            # 3. 构造文本输入
            # 格式: <Radar_Embeds> + <Text_Token_Embeds>
            # 目标文本: "user: ... assistant: Answer"
            
            texts = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                for c in conversations
            ]
            
            tokenized = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(DEVICE)
            
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            
            # 获取文本 Embedding
            inputs_embeds = llm.model.model.embed_tokens(input_ids)
            
            # 拼接: [Radar, Text]
            # [B, 197 + Seq_Len, 3072]
            inputs_embeds = torch.cat([radar_embeds, inputs_embeds], dim=1)
            
            # 构造 Attention Mask
            # [B, 197 + Seq_Len]
            radar_mask = torch.ones((BATCH_SIZE, radar_embeds.shape[1]), device=DEVICE)
            attention_mask = torch.cat([radar_mask, attention_mask], dim=1)
            
            # 构造 Labels
            # -100 忽略计算 Loss
            # 我们希望 LLM 生成整个对话，或者只生成 Assistant 部分
            # 简单起见：忽略 Radar 部分，训练整个 Text 部分
            ignore_labels = torch.full((BATCH_SIZE, radar_embeds.shape[1]), -100, device=DEVICE)
            labels = torch.cat([ignore_labels, input_ids], dim=1)
            # 把 pad token 的 label 设为 -100
            labels[labels == tokenizer.pad_token_id] = -100

            # --- Loss & Backward ---
            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM
            
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRAD_ACCUM
            loop.set_postfix(loss=loss.item() * GRAD_ACCUM)
        
        # 保存每个 Epoch
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch} Loss: {avg_loss}")
        
        save_path = os.path.join(log_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        llm.save_pretrained(save_path) # 保存 LoRA
        torch.save(projector.state_dict(), os.path.join(save_path, "projector.pth")) # 保存 Projector
        
    print("Training Finished!")

if __name__ == "__main__":
    main()