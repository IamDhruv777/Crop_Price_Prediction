"""
Market graph construction module.
"""
import pandas as pd
import numpy as np
import pickle
import torch
from typing import Tuple, Dict, Any
from .neighbors import find_knn_markets, build_adjacency_dict

class MarketGraph:
    def __init__(self, market_df: pd.DataFrame, k: int = 5, max_distance_km: float = 200, sigma: float = 50.0):
        self.market_df = market_df.copy()
        self.k = k
        self.max_distance_km = max_distance_km
        self.sigma = sigma
        self.edges_df = None
        self.adj_dict = None
        
        # Map market_id to node_index for PyTorch
        self.market_ids = sorted(self.market_df['market_id'].unique())
        self.market_to_idx = {m_id: idx for idx, m_id in enumerate(self.market_ids)}
        self.idx_to_market = {idx: m_id for idx, m_id in enumerate(self.market_ids)}
        self.num_nodes = len(self.market_ids)

    def build(self) -> None:
        """Builds the KNN graph."""
        self.edges_df = find_knn_markets(self.market_df, self.k, self.max_distance_km, self.sigma)
        self.adj_dict = build_adjacency_dict(self.edges_df)

    def get_edges(self) -> pd.DataFrame:
        """Returns edges DataFrame."""
        if self.edges_df is None:
            self.build()
        return self.edges_df

    def get_node_features(self, price_df: pd.DataFrame, weather_df: pd.DataFrame, arrival_df: pd.DataFrame, date: Any, commodity: str) -> dict:
        """
        Returns node features for each market at a given date.
        Uses prices at t-1 (not t) to prevent leakage.
        Returns a dict mapping market_id to a feature vector.
        """
        features = {}
        # Convert date to timestamp for comparison if needed
        dt = pd.to_datetime(date)
        
        for m_id in self.market_ids:
            node_feats = []
            
            # Price feature (t-1)
            # Find closest date strictly before dt
            if not price_df.empty and 'market_id' in price_df.columns:
                p_df = price_df[(price_df['market_id'] == m_id) & (price_df['commodity'] == commodity) & (pd.to_datetime(price_df['date']) < dt)]
                if not p_df.empty:
                    # Sort by date descending and get latest
                    latest_price = p_df.sort_values('date', ascending=False).iloc[0]['price']
                    node_feats.append(latest_price)
                else:
                    node_feats.append(0.0)
            else:
                node_feats.append(0.0)
                
            # Weather features (can be at t)
            if not weather_df.empty and 'market_id' in weather_df.columns:
                w_df = weather_df[(weather_df['market_id'] == m_id) & (pd.to_datetime(weather_df['date']) == dt)]
                if not w_df.empty:
                    row = w_df.iloc[0]
                    # Append all numeric features except date, market_id
                    w_feats = [val for col, val in row.items() if col not in ['date', 'market_id'] and isinstance(val, (int, float))]
                    node_feats.extend(w_feats)
                else:
                    # Assume weather features count if available, otherwise just pad
                    node_feats.extend([0.0] * 3) # Temp placeholder count
            
            # Arrival features (t-1)
            if not arrival_df.empty and 'market_id' in arrival_df.columns:
                a_df = arrival_df[(arrival_df['market_id'] == m_id) & (arrival_df['commodity'] == commodity) & (pd.to_datetime(arrival_df['date']) < dt)]
                if not a_df.empty:
                    latest_arrival = a_df.sort_values('date', ascending=False).iloc[0]['volume']
                    node_feats.append(latest_arrival)
                else:
                    node_feats.append(0.0)
                    
            features[m_id] = np.array(node_feats, dtype=np.float32)
            
        return features

    def to_pytorch_geometric(self, node_features: dict = None, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (edge_index, edge_attr) for PyTorch Geometric.
        edge_index: [2, num_edges] tensor
        edge_attr: [num_edges, 1] edge weights
        """
        edges = self.get_edges()
        if edges.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
            return edge_index, edge_attr
            
        sources = [self.market_to_idx[m] for m in edges['source_market_id'].values]
        targets = [self.market_to_idx[m] for m in edges['neighbor_market_id'].values]
        weights = edges['edge_weight'].values
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
        edge_attr = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
        
        return edge_index, edge_attr

    def to_dgl_graph(self, node_features: dict = None):
        """
        Fallback: returns DGL graph if PyTorch Geometric not available.
        """
        try:
            import dgl
            edges = self.get_edges()
            if edges.empty:
                g = dgl.graph(([], []), num_nodes=self.num_nodes)
                return g
                
            sources = [self.market_to_idx[m] for m in edges['source_market_id'].values]
            targets = [self.market_to_idx[m] for m in edges['neighbor_market_id'].values]
            weights = edges['edge_weight'].values
            
            g = dgl.graph((sources, targets), num_nodes=self.num_nodes)
            g.edata['weight'] = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
            
            if node_features is not None:
                # order node features by market_to_idx
                feats = []
                for i in range(self.num_nodes):
                    m_id = self.idx_to_market[i]
                    feats.append(node_features.get(m_id, np.zeros(1))) # fallback zero
                g.ndata['feat'] = torch.tensor(np.array(feats), dtype=torch.float32)
                
            return g
        except ImportError:
            raise ImportError("DGL is not installed.")

    def save(self, path: str) -> None:
        """Saves graph structure."""
        with open(path, 'wb') as f:
            pickle.dump({
                'market_df': self.market_df,
                'k': self.k,
                'max_distance_km': self.max_distance_km,
                'sigma': self.sigma,
                'edges_df': self.edges_df,
                'adj_dict': self.adj_dict,
                'market_to_idx': self.market_to_idx,
                'idx_to_market': self.idx_to_market,
                'num_nodes': self.num_nodes
            }, f)

    @classmethod
    def load(cls, path: str) -> 'MarketGraph':
        """Loads graph."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        graph = cls(data['market_df'], data['k'], data['max_distance_km'], data['sigma'])
        graph.edges_df = data['edges_df']
        graph.adj_dict = data['adj_dict']
        graph.market_to_idx = data['market_to_idx']
        graph.idx_to_market = data['idx_to_market']
        graph.num_nodes = data['num_nodes']
        return graph
