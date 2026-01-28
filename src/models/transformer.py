# Transformer model for sequence classification tasks
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(BaseModel):
    """Transformer model for sequence classification.
    
    Suitable for text classification, time series, or other sequential data.
    Uses self-distillation friendly architecture with dropout and
    configurable depth.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        num_classes: int = 10,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # Use (seq_len, batch, embed_dim) format
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) or (seq_len, batch_size)
            mask: Optional attention mask
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Ensure input is (seq_len, batch_size)
        if x.dim() == 2 and x.size(0) < x.size(1):
            # Assume (batch_size, seq_len) was passed
            x = x.transpose(0, 1)
            
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Create padding mask if not provided
        if mask is None and self.pad_idx is not None:
            # x is (seq_len, batch_size, d_model)
            # Need to check original input for padding
            mask = (x.sum(dim=-1) == 0).transpose(0, 1)  # (batch_size, seq_len)
        
        # Transformer encoding
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling over sequence dimension
        output = output.mean(dim=0)  # (batch_size, d_model)
        
        # Classification
        output = self.dropout(output)
        return self.fc(output)
    
    def get_features(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Extract features before classification layer.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Feature tensor of shape (batch_size, d_model)
        """
        if x.dim() == 2 and x.size(0) < x.size(1):
            x = x.transpose(0, 1)
            
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        if mask is None and self.pad_idx is not None:
            mask = (x.sum(dim=-1) == 0).transpose(0, 1)
            
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        return output.mean(dim=0)


class TransformerForImages(BaseModel):
    """Vision Transformer style model for image classification.
    
    Treats image patches as sequences. Good for CIFAR-10/100 and
    demonstrating self-distillation on transformer architectures.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches.
        
        Args:
            x: Image tensor (B, C, H, W)
            
        Returns:
            Patches (B, N, d_model) where N = num_patches
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d_model)
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Image tensor (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patchify(x)  # (B, N, d_model)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer
        x = self.transformer(x)  # (B, N+1, d_model)
        
        # Use cls token for classification
        x = x[:, 0]  # (B, d_model)
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head.
        
        Args:
            x: Image tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, d_model)
        """
        B = x.shape[0]
        
        x = self.patchify(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        x = x[:, 0]
        return self.norm(x)


def TransformerSmall(
    vocab_size: int = 10000,
    num_classes: int = 10,
    dropout: float = 0.1,
) -> TransformerClassifier:
    """Small transformer for quick experiments."""
    return TransformerClassifier(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=1024,
        num_classes=num_classes,
        dropout=dropout,
    )


def VisionTransformerSmall(
    num_classes: int = 10,
    dropout: float = 0.1,
) -> TransformerForImages:
    """Small vision transformer for CIFAR-10."""
    return TransformerForImages(
        img_size=32,
        patch_size=4,
        d_model=384,
        nhead=6,
        num_layers=6,
        dim_feedforward=1536,
        num_classes=num_classes,
        dropout=dropout,
    )
