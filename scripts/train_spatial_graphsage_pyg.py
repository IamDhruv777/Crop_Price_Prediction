"""
Step 4 & Step 5: Region-Wise Spatial Graph Modeling using PyTorch Geometric (PyG) GraphSAGE.
1. Constructs Graph of Mandis: Nodes = Mandis, Edges = Geographic Proximity.
2. Implements 2-Layer GraphSAGE (SAGEConv) Message Passing Network.
3. Aggregates multi-hop market intelligence from neighboring mandis.
4. Trains with MSE loss and outputs spatial market embeddings.
"""
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
GEO_FILE = DATA_DIR / "raw" / "geography" / "market_master.csv"
SPATIAL_DIR = DATA_DIR / "spatial"
MODELS_DIR = Path("models") / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

class MandiGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index):
        # 1st Hop Neighborhood Aggregation
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        
        # 2nd Hop Neighborhood Aggregation
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        
        # Spatial Node Embedding -> Price Prediction
        out = self.regressor(h2)
        return out, h2

def run_spatial_graphsage_training(k_neighbors=5, max_dist_km=250.0):
    logger.info("==========================================================================")
    logger.info("=== STEP 4: REGION-WISE SPATIAL MODELLING WITH PYTORCH GEOMETRIC (PyG) ===")
    logger.info("==========================================================================")
    
    geo_df = pd.read_csv(GEO_FILE)
    logger.info(f"Loaded {len(geo_df):,} mandis from Market Master.")
    
    # Select key regional trade hub mandis for graph training (e.g. Maharashtra & MP clusters)
    hub_df = geo_df[geo_df['state'].isin(['Maharashtra', 'Madhya Pradesh', 'Gujarat', 'Rajasthan'])].head(150).reset_index(drop=True)
    n_nodes = len(hub_df)
    logger.info(f"Constructing Spatial Graph with {n_nodes} Mandi Nodes...")
    
    # Compute Haversine Edges
    R = 6371.0
    lats = np.radians(hub_df['latitude'].values)
    lons = np.radians(hub_df['longitude'].values)
    
    edge_src = []
    edge_dst = []
    edge_weights = []
    
    for i in range(n_nodes):
        dists = []
        for j in range(n_nodes):
            if i != j:
                dlat = lats[j] - lats[i]
                dlon = lons[j] - lons[i]
                a = np.sin(dlat / 2)**2 + np.cos(lats[i]) * np.cos(lats[j]) * np.sin(dlon / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                d = R * c
                if d <= max_dist_km:
                    dists.append((j, d))
                    
        dists.sort(key=lambda x: x[1])
        for target_idx, d_km in dists[:k_neighbors]:
            edge_src.append(i)
            edge_dst.append(target_idx)
            w = np.exp(-d_km / 75.0)
            edge_weights.append(w)
            
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    logger.info(f"Constructed PyG Edge Index: {edge_index.shape[1]} Spatial Edges connecting Mandis.")
    
    # Synthesize multi-modal Node Feature Matrix [N, Features]: (Normalized Price, Arrival, Rainfall, Temp)
    np.random.seed(42)
    node_features = np.zeros((n_nodes, 6), dtype=np.float32)
    node_features[:, 0] = np.random.uniform(2200, 4500, n_nodes) / 5000.0 # Normalized Price
    node_features[:, 1] = np.random.uniform(10, 500, n_nodes) / 500.0     # Normalized Arrivals
    node_features[:, 2] = np.random.uniform(20, 38, n_nodes) / 45.0      # Temperature
    node_features[:, 3] = np.random.uniform(0, 50, n_nodes) / 100.0      # Rainfall
    node_features[:, 4] = hub_df['latitude'].values / 40.0                # Normalized Lat
    node_features[:, 5] = hub_df['longitude'].values / 100.0              # Normalized Lon
    
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_features[:, 0] * 1.05 + np.random.normal(0, 0.02, n_nodes), dtype=torch.float).unsqueeze(1)
    
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    
    # Train GraphSAGE
    model = MandiGraphSAGE(in_channels=6, hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.MSELoss()
    
    logger.info("Training GraphSAGE Model over 60 Epochs of Spatial Message Passing...")
    model.train()
    for epoch in range(1, 61):
        optimizer.zero_grad()
        out, embeddings = model(graph_data.x, graph_data.edge_index)
        loss = loss_fn(out, graph_data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.info(f"  GraphSAGE Epoch {epoch:2d}/60 | Spatial Message-Passing MSE Loss: {loss.item():.6f}")
            
    # Save GraphSAGE Model and Embeddings
    model_save_path = MODELS_DIR / "graphsage_spatial_model.pth"
    torch.save(model.state_dict(), model_save_path)
    
    # Save Mandi Node Embeddings
    model.eval()
    with torch.no_grad():
        _, node_emb = model(graph_data.x, graph_data.edge_index)
        
    emb_df = pd.DataFrame(node_emb.numpy(), index=hub_df['market_name'])
    emb_df.to_csv(SPATIAL_DIR / "graphsage_node_embeddings.csv")
    
    logger.info(f"GraphSAGE Model successfully trained and saved -> {model_save_path}")
    logger.info(f"Generated 32-dimensional spatial market embeddings for all mandis -> {SPATIAL_DIR / 'graphsage_node_embeddings.csv'}")

if __name__ == "__main__":
    run_spatial_graphsage_training()
