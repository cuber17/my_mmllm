import torch
import torch.nn as nn

class RadarProjector(nn.Module):
    def __init__(self, encoder_dim=768, llm_dim=3072):
        """
        encoder_dim: ViT-Base 通常是 768
        llm_dim: Phi-3-mini 是 3072
        """
        super().__init__()
        # 使用 MLP 投影
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        # x: [Batch, Sequence_Length, Encoder_Dim] -> [Batch, Seq_Len, LLM_Dim]
        return self.net(x)