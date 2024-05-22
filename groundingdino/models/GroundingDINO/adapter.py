import torch
import torch.nn as nn
import math
from .moe import MoE
from torch import Tensor

class LinearAdapter(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 activation=nn.ReLU(inplace=True),
                 ffn_drop=0,
                 num_gate_embed=5,
                 gate_T=2.0,
                 gate_base_scale=0.5,
                 use_gate=True,
                 use_self_kd=True,
                 output_dim=None,
                 **kwargs,
                 ):

        super().__init__()
        if output_dim is None:
            output_dim = embed_dim
        self.adapter_relu = activation
        self.dropout = nn.Dropout(p=ffn_drop)
        self.gate = nn.Embedding(num_gate_embed, embed_dim)
        self.gate_T = gate_T
        self.gate_base_scale = gate_base_scale
        self.use_gate = use_gate
        self.use_self_kd = use_self_kd
        if use_self_kd:
            self.adapter_loss = torch.nn.L1Loss(reduction='mean')
        self.output_dim = output_dim
        self.linear = nn.Linear(embed_dim, output_dim)

        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
    
    def gate_scale(self, x):
        if not self.use_gate:
            return self.gate_base_scale
        B, N, D = x.shape
        gate_weight = self.gate.weight
        x = x / x.norm(dim=-1, keepdim=True) # B N D
        gate_weight = gate_weight / gate_weight.norm(dim=-1, keepdim=True) # 
        max_similarity = torch.max(x.view(B*N, D) @ gate_weight.t(), dim=-1)[0].view(B, N)
        scale = self.gate_base_scale * (self.gate_T * \
            max_similarity).sigmoid().unsqueeze(-1).repeat(1, 1, self.output_dim)
        return scale
    
    def forward(self, x, **kwargs):
        moe_loss = torch.zeros(1).to(x)
        adapter_out = self.linear(x)
        if self.use_self_kd:
            loss_self_kd = self.adapter_loss(x, torch.zeros_like(x))
            moe_loss = moe_loss + loss_self_kd
        return adapter_out * self.gate_scale(x), moe_loss


class TransformerAdapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        nhead=8,
        down_dim=2048,
        ffn_drop=0,
        activation=nn.ReLU(inplace=True),
        normalize_before=False,
        use_self_kd=False,
        output_dim=None,
        **kwargs,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=ffn_drop)
        self.linear1 = nn.Linear(embed_dim, down_dim)
        self.dropout = nn.Dropout(ffn_drop)
        self.linear2 = nn.Linear(down_dim, embed_dim)
        self.project_out = nn.Linear(embed_dim, output_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(ffn_drop)
        self.dropout2 = nn.Dropout(ffn_drop)

        self.normalize_before = normalize_before
        self.nhead = nhead
        self.activation = activation

        self.use_self_kd = use_self_kd
        if use_self_kd:
            self.adapter_loss = torch.nn.L1Loss(reduction='mean')

        with torch.no_grad():
            nn.init.zeros_(self.project_out.weight)
            nn.init.zeros_(self.project_out.bias)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        pos=None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = self.project_out(src)
        
        moe_loss = torch.zeros(1).to(src)
        if self.use_self_kd:
            loss_self_kd = self.adapter_loss(src, torch.zeros_like(src))
            moe_loss = moe_loss + loss_self_kd
        return src, moe_loss
    
    
class Adapter(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 down_dim=64,
                 activation=nn.ReLU(inplace=True),
                 ffn_drop=0,
                 num_gate_embed=5,
                 gate_T=2.0,
                 gate_base_scale=0.5,
                 use_gate=True,
                 use_self_kd=True,
                 output_dim=None,
                 **kwargs,
                 ):
        super().__init__()
        if output_dim is None:
            output_dim = embed_dim
        self.adapter_down = nn.Linear(embed_dim, down_dim)
        self.adapter_relu = activation
        self.adapter_up = nn.Linear(down_dim, output_dim)
        self.dropout = nn.Dropout(p=ffn_drop)
        # self.adapter_out_norm_mean = 0.0
        self.gate = nn.Embedding(num_gate_embed, embed_dim)
        self.gate_T = gate_T
        self.gate_base_scale = gate_base_scale
        self.use_gate = use_gate
        self.use_self_kd = use_self_kd
        if use_self_kd:
            self.adapter_loss = torch.nn.L1Loss(reduction='mean')
        self.output_dim = output_dim

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_down.bias)
            nn.init.zeros_(self.adapter_up.bias)
    
    def gate_scale(self, x):
        if not self.use_gate:
            return self.gate_base_scale
        B, N, D = x.shape
        gate_weight = self.gate.weight
        x = x / x.norm(dim=-1, keepdim=True) # B N D
        gate_weight = gate_weight / gate_weight.norm(dim=-1, keepdim=True) # 
        max_similarity = torch.max(x.view(B*N, D) @ gate_weight.t(), dim=-1)[0].view(B, N)
        scale = self.gate_base_scale * (self.gate_T * \
            max_similarity).sigmoid().unsqueeze(-1).repeat(1, 1, self.output_dim)
        return scale
    
    def forward(self, x, **kwargs):
        moe_loss = torch.zeros(1).to(x)
        adapter_out = self.adapter_up(self.dropout(self.adapter_relu(self.adapter_down(x))))
        if self.use_self_kd:
            loss_self_kd = self.adapter_loss(x, torch.zeros_like(x))
            moe_loss = moe_loss + loss_self_kd
        return adapter_out * self.gate_scale(x), moe_loss


class MoeAdapter(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 down_dim=64,
                 ffn_drop=0,
                 gate_base_scale=0.5,
                 num_experts=10,
                 topk=2,
                 use_self_kd=True,
                 output_dim=None,
                 **kwargs,
                 ):
        super().__init__()
        if output_dim is None:
            output_dim = embed_dim
        self.adapter_moe = MoE(input_size=embed_dim,
                               output_size=output_dim,
                               num_experts=num_experts,
                               hidden_size=down_dim,
                               noisy_gating=True,
                               k=topk,
                               dropout=ffn_drop)
        self.gate_scale = gate_base_scale
        self.use_self_kd = use_self_kd
        if use_self_kd:
            self.adapter_loss = torch.nn.L1Loss(reduction='mean')
        
    
    def forward(self, x, loss_coef=1.0, **kwargs):
        b, n, d = x.shape
        x = x.view(b*n, d)
        x, moe_loss = self.adapter_moe(x, loss_coef=loss_coef)
        x = x.view(b, n, -1)
        if self.use_self_kd:
            loss_self_kd = self.adapter_loss(x, torch.zeros_like(x))
            moe_loss = moe_loss + loss_self_kd
        x = x * self.gate_scale
        return x, moe_loss


zero_value = 1e-8
lan_scale = 0.1
vis_scale = 0.1


class RepZeroLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=False, device=None, dtype=None, down_dim=None) -> None:
        super().__init__()
        self.scaling = nn.parameter.Parameter(torch.ones(1) * lan_scale)
        if down_dim is None:
            down_dim = in_features // 4
        self.freeze_linear = nn.Linear(in_features, out_features, False, device, dtype)
        self.down = nn.Linear(in_features, down_dim, False, device, dtype)
        self.up = nn.Linear(down_dim, out_features, False, device, dtype)
        nn.init.constant_(self.up.weight, val=zero_value)
        nn.init.constant_(self.down.weight, val=zero_value)
        nn.init.constant_(self.freeze_linear.weight, val=0.0)
        
        # self.zero_inter_loss = torch.nn.L1Loss(reduction='mean')
        # self.zero_inter_loss = torch.nn.MSELoss(reduction='mean')
        self.zero_inter_loss = torch.nn.SmoothL1Loss(reduction='mean')

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            branch_output = self.scaling * self.up(self.down(input))
            output = branch_output + self.freeze_linear(input)
            return output, \
                self.zero_inter_loss(branch_output, torch.zeros_like(branch_output)) + \
                    self.zero_inter_loss(output, torch.zeros_like(output))
        else:
            return self.freeze_linear(input), torch.zeros(1).to(input)

    def __rep__(self):
        self.freeze_linear.weight.data = (self.up.weight.data @ self.down.weight.data) * self.scaling \
            + self.freeze_linear.weight.data
        self.scaling = nn.parameter.Parameter(torch.ones(1).to(self.freeze_linear.weight.data) * lan_scale)
        nn.init.constant_(self.up.weight, val=zero_value)
        nn.init.constant_(self.down.weight, val=zero_value)