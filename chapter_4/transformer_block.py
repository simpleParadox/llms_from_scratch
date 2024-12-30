import sys
sys.path.append("D:\llms_from_scratch\\")
from chapter_3.efficient_multi_head_attention import EfficientMultiHeadAttention
from feedforward_with_gelu import FeedForward
from layer_normalization import LayerNorm
import json
import torch.nn as nn
import torch

# with open('gpt_config.json', 'r') as f:
#     cfg = json.load(f)
# cfg = cfg['GPT_CONFIG_124M']


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = EfficientMultiHeadAttention(
            in_dim=cfg["emb_dim"],
            out_dim=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            dropout_value=cfg["drop_rate_att"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])
        
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x # Update the residual / shortcut connection
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x
        

# # Testing the TransformerBlock class.
# torch.manual_seed(123)
# inputs = torch.rand((2, 4, 768))
# block = TransformerBlock(cfg)
# output = block(inputs)
# print("Output shape: ", output.shape) # Should be 2x4x768.