import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))
"""
Neural Network Models for Causal Inference
FT-Transformer + TARNet Architecture
Following: "FT-Transformer: Feature Tokenizer Transformer for Tabular Data" (ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================
# FT-Transformer Components
# =========================================================

class FeatureTokenizer(nn.Module):
    """
    Converts tabular features into token embeddings for Transformer.
    Handles both continuous and categorical features.
    """
    def __init__(self, num_continuous, num_categorical=None, cat_cardinalities=None, 
                 d_token=32, bias=True):
        super().__init__()
        self.num_continuous = num_continuous
        self.num_categorical = num_categorical or 0
        self.d_token = d_token
        
        # Continuous features: each feature gets its own embedding
        if num_continuous > 0:
            self.continuous_embeddings = nn.Parameter(
                torch.Tensor(num_continuous, d_token)
            )
            self.continuous_bias = nn.Parameter(
                torch.Tensor(num_continuous, d_token) if bias else torch.zeros(num_continuous, d_token)
            )
            nn.init.kaiming_uniform_(self.continuous_embeddings, a=math.sqrt(5))
            if bias:
                nn.init.kaiming_uniform_(self.continuous_bias, a=math.sqrt(5))
        
        # Categorical features: embedding table for each categorical feature
        if self.num_categorical > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in (cat_cardinalities or [2] * self.num_categorical)
            ])
        
        # Bias for categorical (if needed)
        if bias and self.num_categorical > 0:
            self.categorical_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(d_token)) for _ in range(self.num_categorical)
            ])
        else:
            self.categorical_bias = []
    
    def forward(self, x_cont, x_cat=None):
        """
        x_cont: (batch, num_continuous) - continuous features
        x_cat: (batch, num_categorical) - categorical features (optional)
        """
        tokens = []
        
        # Process continuous features
        if self.num_continuous > 0:
            # x_cont: (batch, num_cont) -> (batch, num_cont, 1) * (1, num_cont, d_token)
            cont_tokens = x_cont.unsqueeze(-1) * self.continuous_embeddings.unsqueeze(0)
            cont_tokens = cont_tokens + self.continuous_bias.unsqueeze(0)
            tokens.append(cont_tokens)
        
        # Process categorical features
        if x_cat is not None and self.num_categorical > 0:
            cat_tokens = []
            for i, emb in enumerate(self.categorical_embeddings):
                cat_emb = emb(x_cat[:, i])  # (batch, d_token)
                if len(self.categorical_bias) > i:
                    cat_emb = cat_emb + self.categorical_bias[i]
                cat_tokens.append(cat_emb.unsqueeze(1))  # (batch, 1, d_token)
            tokens.append(torch.cat(cat_tokens, dim=1))
        
        # Concatenate all tokens: (batch, num_features, d_token)
        return torch.cat(tokens, dim=1)


class FTTransformerBlock(nn.Module):
    """
    Single Transformer block with Pre-LayerNorm architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation='relu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout_module = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-LN Transformer
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout_module(ff_out)
        
        return x


class FTTransformerEncoder(nn.Module):
    """
    Stack of FT-Transformer blocks.
    """
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FTTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: (batch, num_features, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# =========================================================
# TARNet with FT-Transformer Backbone
# =========================================================

class TARNet_FTTransformer(nn.Module):
    """
    TARNet Architecture with FT-Transformer as feature extractor.
    
    Architecture:
    1. Feature Tokenizer: converts tabular data to tokens
    2. Shared FT-Transformer Encoder: learns representations
    3. Two Hypothesis Heads: one for control (W=0), one for treated (W=1)
    """
    def __init__(self, num_continuous, num_categorical=0, cat_cardinalities=None,
                 d_token=32, num_layers=3, nhead=4, dim_feedforward=128,
                 dropout=0.1, head_hidden_dim=64):
        super().__init__()
        
        self.d_token = d_token
        
        # 1. Feature Tokenizer
        self.tokenizer = FeatureTokenizer(
            num_continuous=num_continuous,
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token
        )
        
        # 2. Shared FT-Transformer Encoder
        self.encoder = FTTransformerEncoder(
            num_layers=num_layers,
            d_model=d_token,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 3. Two Hypothesis Heads (TARNet style)
        # Head 0: Control (W=0)
        self.head_0 = nn.Sequential(
            nn.Linear(d_token, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1)
        )
        
        # Head 1: Treated (W=1)
        self.head_1 = nn.Sequential(
            nn.Linear(d_token, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1)
        )
    
    def encode(self, x_cont, x_cat=None):
        """Return the shared representation used by both outcome heads."""
        tokens = self.tokenizer(x_cont, x_cat)
        encoded = self.encoder(tokens)
        return encoded.mean(dim=1)

    def forward(self, x_cont, x_cat=None, w=None, return_representation=False):
        """
        x_cont: (batch, num_continuous) - continuous features
        x_cat: (batch, num_categorical) - categorical features (optional)
        w: (batch,) - treatment indicator (0 or 1)
        
        Returns:
        - If w is provided: predicted outcome y based on treatment
        - If w is None: returns both head outputs for treatment effect estimation
        """
        representation = self.encode(x_cont, x_cat)
        
        # Get predictions from both heads
        y0_pred = self.head_0(representation)  # Control outcome
        y1_pred = self.head_1(representation)  # Treated outcome
        
        if return_representation:
            return y0_pred, y1_pred, representation
        if w is not None:
            # Select prediction based on treatment
            w = w.unsqueeze(-1).float()
            y_pred = w * y1_pred + (1 - w) * y0_pred
            return y_pred
        else:
            # Return both for treatment effect estimation
            return y0_pred, y1_pred
    
    def predict_effect(self, x_cont, x_cat=None):
        """
        Predict individual treatment effect (ITE/CATE).
        tau(x) = y1(x) - y0(x)
        """
        y0_pred, y1_pred = self.forward(x_cont, x_cat, w=None)
        return y1_pred - y0_pred


# =========================================================
# Simple MLP Baseline (for comparison)
# =========================================================

class TARNet_MLP(nn.Module):
    """
    Simple MLP-based TARNet for baseline comparison.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1, head_hidden_dim=32):
        super().__init__()
        
        # Shared representation layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)
        
        # Two hypothesis heads
        self.head_0 = nn.Sequential(
            nn.Linear(prev_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )
        self.head_1 = nn.Sequential(
            nn.Linear(prev_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )
    
    def forward(self, x, w=None):
        rep = self.shared_net(x)
        y0_pred = self.head_0(rep)
        y1_pred = self.head_1(rep)
        
        if w is not None:
            w = w.unsqueeze(-1).float()
            y_pred = w * y1_pred + (1 - w) * y0_pred
            return y_pred
        else:
            return y0_pred, y1_pred
    
    def predict_effect(self, x):
        y0_pred, y1_pred = self.forward(x, w=None)
        return y1_pred - y0_pred


# =========================================================
# Model Factory
# =========================================================

def create_model(model_type, num_features, num_categorical=0, **kwargs):
    """
    Factory function to create models.
    """
    if model_type == 'ft_transformer':
        return TARNet_FTTransformer(
            num_continuous=num_features - num_categorical,
            num_categorical=num_categorical,
            **kwargs
        )
    elif model_type == 'mlp':
        return TARNet_MLP(input_dim=num_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    print("Testing FT-Transformer + TARNet...")
    
    batch_size = 32
    num_continuous = 25
    num_categorical = 0
    
    model = TARNet_FTTransformer(
        num_continuous=num_continuous,
        d_token=32,
        num_layers=3,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1
    )
    
    x_cont = torch.randn(batch_size, num_continuous)
    w = torch.randint(0, 2, (batch_size,))
    
    # Test forward pass with treatment
    y_pred = model(x_cont, w=w)
    print(f"Prediction shape: {y_pred.shape}")
    
    # Test treatment effect prediction
    tau_pred = model.predict_effect(x_cont)
    print(f"ITE prediction shape: {tau_pred.shape}")
    
    print("✅ Model test passed!")
