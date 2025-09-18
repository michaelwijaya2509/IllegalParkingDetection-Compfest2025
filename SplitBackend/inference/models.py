import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class MultiHeadAttention(nn.Module):

    """ Multi-Head Attention buat temporal modeling """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        batch_size, seq_len = x.size(0), x.size(1)
        residual = x
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Outputnya projection dan residual
        output = self.w_o(attention_output)
        output = self.dropout(output)
        return self.layer_norm(output + residual)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class FeatureAdapter(nn.Module):

    """ Adapter layer buat mapping CNN features ke sequence modeling """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        # x: (B, T, input_dim) atau (B*T, input_dim)
        original_shape = x.shape
        if len(original_shape) == 3:
            B, T, D = original_shape
            x = x.view(B*T, D)
            adapted = self.adapter(x) + self.residual(x)
            return adapted.view(B, T, -1)
        else:
            return self.adapter(x) + self.residual(x)
        
class SequenceHeadLSTM(nn.Module):

    """
    Enhanced Head temporal LSTM dengan Multi-Head Attention, Bidirectional LSTM, dan Residual connections.
    Input :
      feats   : (B, T, D)  -> fitur dari CNN per frame
      lengths : (B,)       -> panjang asli tiap sequence (<= T)
    Output:
      logits  : (B, num_classes)
    """

    def __init__(self, feat_dim: int = 1536, adapter_dim: int = 512, hidden: int = 512, 
                 layers: int = 1, num_classes: int = 2, bidirectional: bool = True, 
                 dropout: float = 0.3, num_attention_heads: int = 8, use_residual: bool = True):
        super().__init__()
        
        # Feature adapter
        self.feature_adapter = FeatureAdapter(feat_dim, adapter_dim, dropout)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=adapter_dim, 
            num_heads=num_attention_heads, 
            dropout=dropout
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=adapter_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if layers > 1 else 0.0),
        )
        
        # Residual connection params
        self.use_residual = use_residual
        if use_residual:
            lstm_out_dim = hidden * (2 if bidirectional else 1)
            self.residual_proj = nn.Linear(adapter_dim, lstm_out_dim)
        
        out_dim = hidden * (2 if bidirectional else 1)
        
        # Enhanced classifier dengan lebih banyak layer
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_attention_mask(self, lengths, max_len):

        """ attention mask untuk variable length sequences """

        batch_size = len(lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=lengths.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        return mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T) untuk broadcasting

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, D), lengths harus di CPU untuk pack
        B, T, D = feats.shape
        
        # Feature adaptation
        adapted_feats = self.feature_adapter(feats)  # (B, T, adapter_dim)
        
        # Multi-head attention dengan mask
        attention_mask = self.create_attention_mask(lengths, T)
        attended_feats = self.attention(adapted_feats, mask=attention_mask)  # (B, T, adapter_dim)
        
        # LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            attended_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (h_n, _) = self.lstm(packed)  # h_n: (num_layers * num_dirs, B, H)
        
        # Unpack untuk residual connection
        if self.use_residual:
            unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            seq_mask = torch.arange(T, device=feats.device).unsqueeze(0) < lengths.unsqueeze(1)
            seq_mask = seq_mask.unsqueeze(-1).float()  # (B, T, 1)
            
            pooled_attended = (attended_feats * seq_mask).sum(dim=1) / lengths.unsqueeze(1).float()
            pooled_attended = self.residual_proj(pooled_attended)  # (B, lstm_out_dim)

        # final hidden state
        if self.lstm.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            h_last = h_n[-1]                               # (B, H)

        # Add residual connection
        if self.use_residual:
            h_last = h_last + pooled_attended

        logits = self.classifier(h_last)                   # (B, num_classes)
        return logits
    

class CNNFeatureExtractor(nn.Module):

    """
    Ekstraktor fitur frame berbasis EfficientNet-B7 (tanpa classifier).
    - Jika input (B, C, H, W) -> output (B, D)
    - Jika input (B, T, C, H, W) -> output (B, T, D)
    D = cnn_feature_size (Eff-B7 = 2560).
    """
    
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()  # keluarkan vektor fitur akhir (B, 1536)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # (B, C, H, W)
            return self.backbone(x)  # (B, D)

        elif x.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feats = self.backbone(x_flat)         # (B*T, D)
            D = feats.shape[1]
            return feats.view(B, T, D)            # (B, T, D)

        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
class CNNLSTMModel(nn.Module):

    """
    Enhanced CNN+LSTM model dengan Multi-Head Attention dan Feature Adapter.
    Input : x (B,T,C,H,W), lengths (B,)
    Output: logits (B,num_classes)
    """

    def __init__(self, num_classes=2, cnn_feature_size=1536, adapter_dim=512,
                 lstm_hidden=512, lstm_layers=1, bidirectional=True, dropout=0.3,
                 num_attention_heads=8, use_residual=True, freeze_backbone=True):
        super().__init__()
        self.extractor = CNNFeatureExtractor(freeze_backbone=freeze_backbone)
        self.head = SequenceHeadLSTM(
            feat_dim=cnn_feature_size,
            adapter_dim=adapter_dim,
            hidden=lstm_hidden,
            layers=lstm_layers,
            num_classes=num_classes,
            bidirectional=bidirectional,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            use_residual=use_residual
        )

    def forward(self, x_btchw: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x_btchw)            # (B, T, D)
        logits = self.head(feats, lengths)         # (B, num_classes)
        return logits
    