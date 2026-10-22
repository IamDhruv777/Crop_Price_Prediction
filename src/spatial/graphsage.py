"""
GraphSAGE for spatial market embedding.
Aggregates information from neighboring markets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    
    # Manual implementation if PyG is missing
    class SAGEConv(nn.Module):
        def __init__(self, in_channels, out_channels, aggr='mean'):
            super().__init__()
            self.aggr = aggr
            self.lin_l = nn.Linear(in_channels, out_channels)
            self.lin_r = nn.Linear(in_channels, out_channels)
            
        def forward(self, x, edge_index):
            num_nodes = x.size(0)
            row, col = edge_index
            
            out = torch.zeros((num_nodes, x.size(1)), device=x.device)
            if edge_index.size(1) > 0:
                if self.aggr == 'mean':
                    out.index_add_(0, row, x[col])
                    deg = torch.zeros(num_nodes, device=x.device)
                    ones = torch.ones(edge_index.size(1), device=x.device)
                    deg.index_add_(0, row, ones)
                    deg = deg.clamp(min=1)
                    out = out / deg.unsqueeze(1)
                elif self.aggr == 'max':
                    out.index_add_(0, row, x[col])
            
            out = self.lin_l(x) + self.lin_r(out)
            return F.normalize(out, p=2, dim=-1)


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, aggr: str = 'mean'):
        super().__init__()
        self.conv = SAGEConv(in_features, out_features, aggr=aggr)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        return x


class GraphSAGEModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(in_features, hidden_features))
        
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_features, hidden_features))
            
        if num_layers > 1:
            self.layers.append(GraphSAGELayer(hidden_features, out_features))
            
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEForecaster(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 64, out_features: int = 32, num_layers: int = 2):
        super().__init__()
        self.sage = GraphSAGEModel(in_features, hidden_features, out_features, num_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(out_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, 3) # predicted_price, lower_bound, upper_bound
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index, edge_attr=None):
        emb = self.sage(x, edge_index, edge_attr)
        preds = self.predictor(emb)
        return preds

    def fit(self, graph, node_feature_sequences: dict, targets: dict, val_data=None, epochs=50):
        self.train()
        edge_index, edge_attr = graph.to_pytorch_geometric(device='cpu')
        dates = sorted(list(node_feature_sequences.keys()))
        
        # Get actual in_channels since first layer might be dynamically sized or explicit
        in_dim = None
        
        for epoch in range(epochs):
            for date in dates:
                feats = node_feature_sequences[date]
                targs = targets.get(date, {})
                if not targs: continue
                
                x = []
                y = []
                mask = []
                
                # deduce in_dim
                if in_dim is None:
                    for v in feats.values():
                        in_dim = len(v)
                        break
                if in_dim is None:
                    continue
                
                for i in range(graph.num_nodes):
                    m_id = graph.idx_to_market[i]
                    x.append(feats.get(m_id, np.zeros(in_dim)))
                    if m_id in targs:
                        y.append(targs[m_id])
                        mask.append(True)
                    else:
                        y.append(0.0)
                        mask.append(False)
                        
                x = torch.tensor(np.array(x), dtype=torch.float32)
                y = torch.tensor(np.array(y), dtype=torch.float32)
                mask = torch.tensor(mask, dtype=torch.bool)
                
                if not mask.any(): continue
                    
                self.optimizer.zero_grad()
                preds = self(x, edge_index, edge_attr)
                loss = self.criterion(preds[mask, 0], y[mask])
                loss.backward()
                self.optimizer.step()

    def predict(self, graph, node_features: dict, target_market_id: str) -> dict:
        self.eval()
        with torch.no_grad():
            edge_index, edge_attr = graph.to_pytorch_geometric(device='cpu')
            
            in_dim = None
            for v in node_features.values():
                in_dim = len(v)
                break
            if in_dim is None: in_dim = 1
                
            x = []
            for i in range(graph.num_nodes):
                m_id = graph.idx_to_market[i]
                x.append(node_features.get(m_id, np.zeros(in_dim)))
                
            x = torch.tensor(np.array(x), dtype=torch.float32)
            preds = self(x, edge_index, edge_attr)
            
            idx = graph.market_to_idx.get(target_market_id)
            if idx is None:
                return {'predicted_price': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0}
                
            pred_vals = preds[idx].numpy()
            return {
                'predicted_price': float(pred_vals[0]),
                'lower_bound': float(pred_vals[1]),
                'upper_bound': float(pred_vals[2])
            }

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, in_features: int, hidden_features: int = 64, out_features: int = 32, num_layers: int = 2) -> 'GraphSAGEForecaster':
        model = cls(in_features, hidden_features, out_features, num_layers)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
