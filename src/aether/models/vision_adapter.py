import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VisionAdapter(nn.Module):
    """
    Project Aether Visual Encoder.
    Converts images into a sequence of embeddings that the Transformer can process.
    """
    def __init__(self, embed_dim=768, num_visual_tokens=64):
        super().__init__()
        # Load a pre-trained Vision Transformer
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # Remove the classification head
        self.backbone.heads = nn.Identity()
        
        # ViT-B/16 output dim is 768. 
        # We use a projection layer to map this to our internal Transformer's d_model
        self.projection = nn.Linear(768, embed_dim)
        
        # Perceiver-style Resampler (Simplified)
        # Instead of using all patches (e.g., 196), we learn a fixed number of tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_visual_tokens, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        with torch.no_grad():
            features = self.backbone._process_input(x)
            n = features.shape[0]
            # Expand the patch and class tokens
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            features = torch.cat([batch_class_token, features], dim=1)
            features = self.backbone.encoder(features)
            # features: [batch, 197, 768]
            
        projected = self.projection(features) # [batch, 197, embed_dim]
        
        # Resample to fixed num_visual_tokens using cross-attention
        queries = self.query_tokens.expand(projected.shape[0], -1, -1)
        visual_tokens, _ = self.cross_attn(queries, projected, projected)
        
        return visual_tokens # [batch, num_visual_tokens, embed_dim]
