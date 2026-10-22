"""
LSTM sequence model for time-series price forecasting using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple

def create_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, seq_len: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    CRITICAL: sequences must be created in time order
    No shuffling before sequence creation
    seq[i] uses df[i:i+seq_len], target[i] = df[i+seq_len+horizon-1][target_col]
    """
    data_X = df[feature_cols].values
    data_y = df[target_col].values
    
    sequences = []
    targets = []
    
    for i in range(len(df) - seq_len - horizon + 1):
        seq = data_X[i:(i + seq_len)]
        target = data_y[i + seq_len + horizon - 1]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets)

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

class LSTMModel:
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMNet(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }

    def fit(self, X_sequences: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, config: dict = None) -> None:
        if config is None:
            config = {}
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        lr = config.get('lr', 1e-3)
        patience = config.get('patience', 10)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(dataloader.dataset)
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_output = self.model(X_val_tensor)
                    val_loss = criterion(val_output, y_val_tensor).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_lstm.pt')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(torch.load('best_lstm.pt'))
                    break

    def predict(self, X_sequences: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor)
        return preds.cpu().numpy()

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=20) -> dict:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter tuning.")

        def objective(trial):
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            
            model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            model.fit(X_train, y_train, X_val, y_val, config={'epochs': 30, 'lr': lr, 'patience': 5})
            
            preds = model.predict(X_val)
            rmse = np.sqrt(np.mean((preds - y_val)**2))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

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
        instance = cls(input_size=config['input_size'], hidden_size=config['hidden_size'], 
                       num_layers=config['num_layers'], dropout=config['dropout'])
        instance.model.load_state_dict(state['state_dict'])
        return instance
