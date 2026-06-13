import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask.unsqueeze(1)  # (batch, 1, q_len, k_len) broadcasts over heads
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        even_i      = torch.arange(0, d_model, 2).float()
        denominator = torch.pow(10000, even_i / d_model)
        position    = torch.arange(max_sequence_length).reshape(max_sequence_length, 1)
        even_PE     = torch.sin(position / denominator)
        odd_PE      = torch.cos(position / denominator)
        stacked     = torch.stack([even_PE, odd_PE], dim=2)
        PE          = torch.flatten(stacked, start_dim=1, end_dim=2)
        # buffer: moves with .to(device), saved in state_dict, never trained
        self.register_buffer('PE', PE)

    def forward(self, sequence_length):
        return self.PE[:sequence_length]


class TokenEmbedding(nn.Module):
    """Embeds pre-tokenized id tensors (batch, seq_len) — tokenization lives in the data pipeline."""
    def __init__(self, vocab_size, d_model, max_sequence_length, drop_prob=0.1):
        super().__init__()
        self.d_model          = d_model
        self.embedding        = nn.Embedding(vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout          = nn.Dropout(p=drop_prob)

    def forward(self, token_ids):
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.dropout(x + self.position_encoder(token_ids.size(1)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.qkv_layer    = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta  = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var  = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std  = (var + self.eps).sqrt()
        y    = (inputs - mean) / std
        out  = self.gamma * y + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1  = nn.Linear(d_model, hidden)
        self.linear2  = nn.Linear(hidden, d_model)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1     = LayerNormalization(parameters_shape=[d_model])
        self.dropout1  = nn.Dropout(p=drop_prob)
        self.ffn       = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2     = LayerNormalization(parameters_shape=[d_model])
        self.dropout2  = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, vocab_size):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model, max_sequence_length, drop_prob)
        self.layers = SequentialEncoder(
            *[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(self, token_ids, self_attention_mask):
        x = self.embedding(token_ids)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.kv_layer     = nn.Linear(d_model, 2 * d_model)
        self.q_layer      = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask):
        # x: encoder output (keys/values), y: decoder stream (queries) —
        # their sequence lengths are independent
        batch_size, src_length, d_model = x.size()
        tgt_length = y.size(1)
        kv = self.kv_layer(x)
        q  = self.q_layer(y)
        kv = kv.reshape(batch_size, src_length, self.num_heads, 2 * self.head_dim)
        q  = q.reshape(batch_size, tgt_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q  = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, tgt_length, d_model)
        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.self_attention            = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1               = LayerNormalization(parameters_shape=[d_model])
        self.dropout1                  = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2               = LayerNormalization(parameters_shape=[d_model])
        self.dropout2                  = nn.Dropout(p=drop_prob)
        self.ffn                       = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3               = LayerNormalization(parameters_shape=[d_model])
        self.dropout3                  = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y
        y  = self.self_attention(y, mask=self_attention_mask)
        y  = self.dropout1(y)
        y  = self.layer_norm1(y + _y)

        _y = y
        y  = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y  = self.dropout2(y)
        y  = self.layer_norm2(y + _y)

        _y = y
        y  = self.ffn(y)
        y  = self.dropout3(y)
        y  = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, vocab_size):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model, max_sequence_length, drop_prob)
        self.layers = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(self, x, token_ids, self_attention_mask, cross_attention_mask):
        y = self.embedding(token_ids)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 src_vocab_size,
                 tgt_vocab_size):
        super().__init__()
        # Encoder = English (source), Decoder = Nepali (target)
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                               max_sequence_length, src_vocab_size)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                               max_sequence_length, tgt_vocab_size)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src_ids,
                tgt_ids,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None):
        x   = self.encoder(src_ids, encoder_self_attention_mask)
        out = self.decoder(x, tgt_ids, decoder_self_attention_mask, decoder_cross_attention_mask)
        out = self.linear(out)
        return out
