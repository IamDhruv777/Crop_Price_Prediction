import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "features" / "engineered_features.csv"
MODELS_DIR = Path("models") / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, num_layers=2, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def run_deep_learning():
    logger.info("=== INITIALIZING HEAVY DEEP LEARNING (LSTM) ===")
    
    if not FEATURES_FILE.exists():
        logger.error("Features file not found! Run preprocessing first.")
        return
        
    df = pd.read_csv(FEATURES_FILE)
    df = df.sort_values('date').dropna()
    
    # Target: Predict 7 days out
    df['target'] = df.groupby('market')['modal_price'].shift(-7)
    df = df.dropna(subset=['target'])
    
    # Grab engineered columns correctly (they look like 'modal_price_lag_1', etc.)
    ignore_cols = ['target', 'date', 'market', 'arrival_date', 'state', 'district', 'commodity', 'variety', 'grade', 'is_outlier']
    feature_cols = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]
    
    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(df[feature_cols])
    y = scaler_y.fit_transform(df[['target']])
    
    # Reshape for LSTM (Samples, TimeSteps, Features)
    X_seq = X.reshape((X.shape[0], 1, X.shape[1]))
    
    # Chronological Split
    train_size = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X_seq[:train_size])
    y_train = torch.FloatTensor(y[:train_size])
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # NO SHUFFLING - Time Series!
    
    model = LSTMModel(input_size=len(feature_cols))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50 
    logger.info(f"Starting Training for {epochs} Epochs on {len(X_train)} sequences...")
    logger.info("This process is computationally heavy and will take time.")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f'Epoch {epoch:3} | Loss: {total_loss/len(train_loader):.4f}')
            
    logger.info("Deep Learning Training Complete!")
    torch.save(model.state_dict(), MODELS_DIR / "lstm_heavy_model.pth")
    logger.info("Model saved to models/trained/lstm_heavy_model.pth")

if __name__ == "__main__":
    run_deep_learning()
