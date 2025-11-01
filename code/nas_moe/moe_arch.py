import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from nni.nas.space import model_context
from nni.nas.hub.pytorch import DARTS as DartsSpace


class MoE(nn.Module):
    def __init__(
        self,
        expert_architectures: List[Dict],
        SearchSpace,
        input_size: int,
        top_k: int = 1,
        hidden_size: int = 128,
        dataset: str = 'cifar',
        width: int = 16,
        num_cells: int = 8,
        permutation = None
    ):
        super(MoE, self).__init__()
        
        self.num_experts = len(expert_architectures)
        self.top_k = min(top_k, self.num_experts)
        self.input_size = input_size
        
        self.experts = nn.ModuleList()
        for exported_arch in expert_architectures:
            with model_context(exported_arch):
                if permutation is not None:
                    expert = SearchSpace(
                        width=width,
                        num_cells=num_cells,
                        dataset=dataset,
                        permutation=permutation
                    )
                else:
                    expert = SearchSpace(
                        width=width,
                        num_cells=num_cells,
                        dataset=dataset,
                    )
            self.experts.append(expert)
        
        self.router = nn.Sequential(
            nn.Flatten(),                  # ← здесь
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_experts)
        )

        
        self.noise_std = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x_flat = x.view(batch_size, -1)
        
        router_logits = self.router(x_flat)
        
        if self.training:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, 
            self.top_k, 
            dim=-1
        )
        
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        expert_outputs = []
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)
            
            single_expert_output = self.experts[0](x[:1])
            batch_output = torch.zeros(batch_size, *single_expert_output.shape[1:], device=x.device, dtype=single_expert_output.dtype)
            
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_out = self.experts[expert_id](expert_input)
                    batch_output[mask] = expert_out
            
            expert_outputs.append(batch_output * expert_weight)
        
        final_output = sum(expert_outputs)
        
        return final_output
    
    def get_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        router_logits = self.router(x_flat)
        return F.softmax(router_logits, dim=-1)
    
    def compute_load_balancing_loss(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = self.get_routing_weights(x)
        
        expert_usage = routing_weights.mean(dim=0)
        
        mean_usage = expert_usage.mean()
        std_usage = expert_usage.std()
        cv_loss = std_usage / (mean_usage + 1e-10)
        
        return cv_loss
