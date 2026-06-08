"""Graph Neural Network fraud detector."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from typing import Dict, List

class FraudGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int = 64, num_layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList([SAGEConv((-1, -1), hidden_channels) for _ in range(num_layers)])
        self.classifier = torch.nn.Linear(hidden_channels, 2)  # binary: fraud/legit

    def forward(self, x_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
        x = x_dict["account"]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index_dict.get(("account", "transfers", "account"), torch.zeros((2, 0), dtype=torch.long))))
        return self.classifier(x)

def build_transaction_graph(transactions: List[Dict]) -> HeteroData:
    """Build PyG HeteroData from transaction records."""
    data = HeteroData()
    # Collect unique account IDs
    accounts = list(set([t["sender_id"] for t in transactions] + [t["receiver_id"] for t in transactions]))
    acc_idx = {a: i for i, a in enumerate(accounts)}
    # Node features: amount stats, velocity, age
    import numpy as np
    n = len(accounts)
    data["account"].x = torch.randn(n, 16)  # Replace with real features
    # Transfer edges
    edges = [(acc_idx[t["sender_id"]], acc_idx[t["receiver_id"]]) for t in transactions]
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data["account", "transfers", "account"].edge_index = ei
        data["account", "transfers", "account"].edge_attr = torch.tensor(
            [[t["amount"], t.get("hour", 12)] for t in transactions], dtype=torch.float)
    return data
