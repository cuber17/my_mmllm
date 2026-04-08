import argparse
import datetime
import logging
import os

import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from src.llm.dataset import MMLLMDataset
from src.llm.projector import RadarProjector


MODEL_PRESETS = {
    "phi3mini": {
        "local_dir": "Phi-3-mini-4k-instruct",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "phi35": {
        "local_dir": "Phi-3.5-mini-instruct",
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "phi4mini": {
        "local_dir": "Phi-4-mini-instruct",
        "repo_id": "microsoft/Phi-4-mini-instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "qwen25_3b": {
        "local_dir": "Qwen2.5-3B-Instruct",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "gemma2_2b": {
        "local_dir": "gemma-2-2b-it",
        "repo_id": "google/gemma-2-2b-it",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "llama31_8b": {
        "local_dir": "Llama-3.1-8B-Instruct",
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 training script for LLM comparison experiments")
    parser.add_argument("--model_key", type=str, required=True, choices=list(MODEL_PRESETS.keys()))

    parser.add_argument("--project_root", type=str, default="/root/jyz/my_mmLLM")
    parser.add_argument("--model_root", type=str, default="./huggingface")
    parser.add_argument("--radar_encoder_path", type=str, default="logs/clip_20260406_085718/radar_encoder_only.pth")

    parser.add_argument("--train_json", type=str, default="/root/jyz/my_mmLLM/processed_dataset/train.json")
    parser.add_argument("--data_root", type=str, default="/root/jyz/my_mmLLM/processed_dataset/")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_proj", type=float, default=1e-3)
    parser.add_argument("--lr_llm", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()


def resolve_model_path(model_root: str, model_key: str) -> str:
    local_path = os.path.join(model_root, MODEL_PRESETS[model_key]["local_dir"])
    if os.path.isdir(local_path):
        return local_path
    return MODEL_PRESETS[model_key]["repo_id"]


def main() -> None:
    args = build_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.project_root, "logs", f"stage2_{args.model_key}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, "train.log"), level=logging.INFO)

    print(f"Logging to: {log_dir}")

    # 1) Load frozen radar encoder
    radar_ckpt = args.radar_encoder_path
    if not os.path.isabs(radar_ckpt):
        radar_ckpt = os.path.join(args.project_root, radar_ckpt)

    print(">>> Loading Radar Encoder (Frozen)...")
    radar_encoder = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(radar_ckpt, map_location="cpu")
    state = {k.replace("radar_encoder.", ""): v for k, v in state.items()}
    msg = radar_encoder.load_state_dict(state, strict=False)
    print(f"Radar encoder load msg: {msg}")

    radar_encoder.to(args.device).eval()
    for p in radar_encoder.parameters():
        p.requires_grad = False

    # 2) Load LLM + LoRA
    model_path = resolve_model_path(args.model_root, args.model_key)
    print(f">>> Loading LLM [{args.model_key}] from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=MODEL_PRESETS[args.model_key]["lora_targets"],
    )
    llm = get_peft_model(llm, lora_cfg)
    llm.print_trainable_parameters()

    # 3) Projector uses dynamic hidden size from current LLM
    llm_hidden_size = int(getattr(llm.config, "hidden_size"))
    print(f"LLM hidden size: {llm_hidden_size}")

    projector = RadarProjector(encoder_dim=768, llm_dim=llm_hidden_size).to(args.device)

    # 4) Dataset
    dataset = MMLLMDataset(json_file=args.train_json, root_dir=args.data_root)

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        conversations = [item["conversation"] for item in batch]
        return pixel_values, conversations

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": projector.parameters(), "lr": args.lr_proj},
            {"params": llm.parameters(), "lr": args.lr_llm},
        ]
    )

    print(">>> Start Training...")
    projector.train()
    llm.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, (pixel_values, conversations) in enumerate(pbar):
            bs = pixel_values.size(0)
            pixel_values = pixel_values.to(args.device, dtype=torch.bfloat16)

            with torch.no_grad():
                radar_feats = radar_encoder.forward_features(pixel_values.float())

            radar_embeds = projector(radar_feats.to(args.device))
            radar_embeds = radar_embeds.to(dtype=torch.bfloat16)

            texts = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                for c in conversations
            ]
            tokenized = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(args.device)

            input_ids = tokenized.input_ids
            text_attention_mask = tokenized.attention_mask

            text_embeds = llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([radar_embeds, text_embeds], dim=1)

            radar_mask = torch.ones((bs, radar_embeds.shape[1]), device=args.device, dtype=text_attention_mask.dtype)
            attention_mask = torch.cat([radar_mask, text_attention_mask], dim=1)

            ignore_labels = torch.full((bs, radar_embeds.shape[1]), -100, device=args.device, dtype=input_ids.dtype)
            labels = torch.cat([ignore_labels, input_ids], dim=1)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            should_step = ((step + 1) % args.grad_accum == 0) or (step + 1 == len(dataloader))
            if should_step:
                optimizer.step()
                optimizer.zero_grad()

            step_loss = float(loss.item() * args.grad_accum)
            total_loss += step_loss
            pbar.set_postfix(loss=step_loss)

        avg_loss = total_loss / max(1, len(dataloader))
        logging.info(f"Epoch {epoch} Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        save_dir = os.path.join(log_dir, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        llm.save_pretrained(save_dir)
        torch.save(projector.state_dict(), os.path.join(save_dir, "projector.pth"))

    print("Training Finished!")


if __name__ == "__main__":
    main()
