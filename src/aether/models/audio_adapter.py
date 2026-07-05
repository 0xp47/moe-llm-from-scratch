import torch
import torch.nn as nn

class AudioAdapter(nn.Module):
    """
    Project Aether Audio Encoder.
    Converts raw audio waveforms into a sequence of embeddings.
    Uses a simplified 1D ConvNet (similar to early layers of Wav2Vec) 
    followed by a projection.
    """
    def __init__(self, embed_dim=768, num_audio_tokens=32):
        super().__init__()
        # Input: [Batch, 1, Time]
        # Simple feature extractor for raw audio
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, bias=False),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=2, bias=False),
            nn.GELU()
        )
        
        # Audio tokens are often long, so we pool or resample them
        # Here we use an Adaptive Average Pool to force a fixed number of tokens
        # In a real system (Gemini), this would be a Conformer or AST
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_audio_tokens)
        
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, audio):
        # audio: [batch, 1, samples] (e.g. 1 sec at 16khz = 16000 samples)
        
        features = self.feature_extractor(audio) 
        # features: [batch, embed_dim, reduced_time]
        
        # Pool to fixed size
        features = self.adaptive_pool(features)
        # features: [batch, embed_dim, num_audio_tokens]
        
        # Transpose for Transformer [batch, seq_len, embed_dim]
        features = features.transpose(1, 2)
        
        return self.projection(features)
