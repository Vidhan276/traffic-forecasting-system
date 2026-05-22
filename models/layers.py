import torch
import torch.nn as nn
import torch.nn.functional as F

class PurePyTorchGATConv(nn.Module):
    """
    Pure PyTorch GATConv layer that does not depend on torch_geometric.
    It matches the parameter shapes and behavior of PyG GATConv.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, bias=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_channels if concat else out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        num_nodes = x.size(0)
        
        # 1. Project input
        x_lin = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # 2. Add self-loops to edge_index (since GATConv defaults to add_self_loops=True)
        loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
        
        # 3. Calculate attention inputs
        alpha_src = (x_lin * self.att_src).sum(dim=-1)  # [num_nodes, heads]
        alpha_dst = (x_lin * self.att_dst).sum(dim=-1)  # [num_nodes, heads]
        
        u = edge_index_with_loops[0]
        v = edge_index_with_loops[1]
        
        alpha = alpha_src[u] + alpha_dst[v]  # [num_edges_with_loops, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # 4. Softmax over incoming edges for each target node v
        max_alpha = torch.zeros(num_nodes, self.heads, device=alpha.device, dtype=alpha.dtype)
        max_alpha.fill_(-9999.0)
        max_alpha = max_alpha.scatter_reduce(0, v.unsqueeze(-1).expand(-1, self.heads), alpha, reduce="amax", include_self=False)
        
        alpha_exp = torch.exp(alpha - max_alpha[v])
        
        sum_exp = torch.zeros(num_nodes, self.heads, device=alpha.device, dtype=alpha.dtype)
        sum_exp = sum_exp.scatter_reduce(0, v.unsqueeze(-1).expand(-1, self.heads), alpha_exp, reduce="sum", include_self=False)
        
        # Attention coefficients
        attn = alpha_exp / (sum_exp[v] + 1e-16)  # [num_edges_with_loops, heads]
        
        # 5. Aggregate message: out_i = sum_{j} attn_{i,j} * x_j
        msg = attn.unsqueeze(-1) * x_lin[u]  # [num_edges_with_loops, heads, out_channels]
        
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
        v_expanded = v.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_channels)
        out = out.scatter_reduce(0, v_expanded, msg, reduce="sum", include_self=False)
        
        # 6. Concat or average heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        # 7. Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out


class PurePyTorchGCNConv(nn.Module):
    """
    Pure PyTorch GCNConv layer that does not depend on torch_geometric.
    It matches the parameter shapes and behavior of PyG GCNConv.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False, add_self_loops=True, bias=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        num_nodes = x.size(0)
        
        # 1. Project input
        x_lin = self.lin(x)
        
        # 2. Add self-loops
        if self.add_self_loops:
            loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
        else:
            edge_index_with_loops = edge_index
            
        u = edge_index_with_loops[0]
        v = edge_index_with_loops[1]
        
        # 3. Compute degree normalization coefficients
        deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        ones = torch.ones(v.size(0), dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, v, ones)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        norm = deg_inv_sqrt[u] * deg_inv_sqrt[v]  # [num_edges]
        
        # 4. Message aggregation
        msg = norm.unsqueeze(-1) * x_lin[u]  # [num_edges, out_channels]
        
        out = torch.zeros(num_nodes, self.out_channels, dtype=x.dtype, device=x.device)
        v_expanded = v.unsqueeze(-1).expand(-1, self.out_channels)
        out.scatter_add_(0, v_expanded, msg)
        
        # 5. Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out
