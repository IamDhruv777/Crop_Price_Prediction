"""
Time-series Transformer for crop price forecasting.
Uses causal masking to prevent future information leakage.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_size=1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, output_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = self.input_linear(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        output = self.transformer_encoder(x, mask=mask)
        output = output[:, -1, :] # Take the last element
        return self.output_linear(output).squeeze(-1)

class TransformerModel:
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeSeriesTransformer(input_size, d_model, nhead, num_layers, dropout=dropout).to(self.device)
        self.config = {
            'input_size': input_size,
            'd_model': d_model,
            'nhead': nhead,
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
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
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
                    torch.save(self.model.state_dict(), 'best_transformer.pt')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(torch.load('best_transformer.pt'))
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
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            
            model = TransformerModel(input_size=X_train.shape[2], d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
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
        instance = cls(input_size=config['input_size'], d_model=config['d_model'], 
                       nhead=config['nhead'], num_layers=config['num_layers'], dropout=config['dropout'])
        instance.model.load_state_dict(state['state_dict'])
        return instance
