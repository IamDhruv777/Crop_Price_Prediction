"""
Spatial-Temporal model: GraphSAGE + GRU.
Combines spatial market graph embeddings with temporal sequence modeling.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    SAGEConv = None

class SpatialTemporalModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_temporal_features, gru_hidden, output_size=1):
        super(SpatialTemporalModel, self).__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for SpatialTemporalModel")
            
        self.sage = SAGEConv(num_node_features, hidden_dim)
        
        # Temporal GRU takes the concatenated spatial embedding and temporal features
        self.gru = nn.GRU(hidden_dim + num_temporal_features, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, output_size)

    def forward(self, x_node, edge_index, x_temporal):
        """
        x_node: [num_nodes, node_features]
        edge_index: [2, num_edges]
        x_temporal: [batch_size, seq_len, temporal_features]
        """
        # 1. Spatial Embedding
        spatial_emb = self.sage(x_node, edge_index) # [num_nodes, hidden_dim]
        spatial_emb = torch.relu(spatial_emb)
        
        # For this simplified model, we pool here to get a global spatial feature:
        global_spatial = spatial_emb.mean(dim=0) # [hidden_dim]
        
        # Expand spatial features for temporal sequences
        batch_size, seq_len, _ = x_temporal.size()
        global_spatial = global_spatial.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # 2. Concatenate
        combined = torch.cat([global_spatial, x_temporal], dim=-1) # [batch_size, seq_len, hidden_dim + temp_feat]
        
        # 3. Temporal Modeling
        out, _ = self.gru(combined)
        out = out[:, -1, :]
        
        # 4. Output
        return self.fc(out).squeeze(-1)

class SpatialTemporalForecaster:
    def __init__(self, num_node_features, hidden_dim=32, num_temporal_features=10, gru_hidden=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpatialTemporalModel(num_node_features, hidden_dim, num_temporal_features, gru_hidden).to(self.device)
        self.config = {
            'num_node_features': num_node_features,
            'hidden_dim': hidden_dim,
            'num_temporal_features': num_temporal_features,
            'gru_hidden': gru_hidden
        }

    def fit(self, x_node: np.ndarray, edge_index: np.ndarray, X_sequences: np.ndarray, y: np.ndarray, config: dict = None) -> None:
        if config is None:
            config = {}
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        lr = config.get('lr', 1e-3)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        x_node_tensor = torch.tensor(x_node, dtype=torch.float32).to(self.device)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(x_node_tensor, edge_index_tensor, batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, x_node: np.ndarray, edge_index: np.ndarray, X_sequences: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_node_tensor = torch.tensor(x_node, dtype=torch.float32).to(self.device)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)
            preds = self.model(x_node_tensor, edge_index_tensor, X_tensor)
        return preds.cpu().numpy()

    def save(self, path: str) -> None:
        state = {
            'state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str):
        state = torch.load(path, map_location=torch.device('cpu'))
        config = state['config']
        instance = cls(num_node_features=config['num_node_features'], hidden_dim=config['hidden_dim'], 
                       num_temporal_features=config['num_temporal_features'], gru_hidden=config['gru_hidden'])
        instance.model.load_state_dict(state['state_dict'])
        return instance
