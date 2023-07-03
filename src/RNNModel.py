import torch

from typing import Any
from tqdm.notebook import tqdm

class RNNModel(torch.nn.Module):
    """Class for RNN model"""
    def __init__(self, 
                 input_size:int, 
                 hidden_size: int=128, 
                 batch_size: int=512, 
                 num_epochs: int=10,
                 num_layers: int=1,
                 lr: float=1e-3):
        super().__init__()
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers


        self.rnn = torch.nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True
                                )
        # expected input format: batch, num of lags, num of features -> batch, 5, 1
        self.linear = torch.nn.Linear(hidden_size, 1)

        # selecting device depending on availability
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.to(device=self.device)

        self.losses = [] # saving losses as class attribute for logging
        self.optimiser = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        """Applies network layers to the passed data"""
        hidden_state = torch.zeros(self.num_layers, token.shape[0], self.hidden_size)
        hidden_state = hidden_state.to(device=self.device)
        out_token, _ = self.rnn(token, hidden_state)
        output = self.linear(out_token[:,-1,:])
        return output
    
    def _train_epoch(self, X: torch.Tensor, y: torch.Tensor):
        """For training the network for 1 epoch"""
        X = X.to(device=self.device)
        y = y.to(device=self.device)

        for ix in tqdm(range(0, len(X), self.batch_size)):
            batch_X = X[ix : ix + self.batch_size]; batch_y = y[ix : ix + self.batch_size]
            rnn_output = self.forward(token=batch_X)
        
            loss = torch.nn.functional.mse_loss(input=rnn_output.flatten(), target=batch_y.flatten())
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
                
            self.losses.append(loss.cpu().detach().numpy())

    def _adjust_type_dims_device(self, x: Any, 
                                 target_dim: int) -> torch.Tensor:
        """Types and shape adjustment, used when non tensors are passed to the model"""
        if not type(x) == torch.Tensor:
            x = torch.Tensor(x).float()
        if len(x.shape) < target_dim:
            x = x.unsqueeze(-1)
        if x.device != self.device:
            x = x.to(device=self.device)
        return x

    def fit(self, X, y):
        """Full fitting of the model"""
        y = self._adjust_type_dims_device(x=y, target_dim=2)
        X = self._adjust_type_dims_device(x=X, target_dim=3)

        for _ in tqdm(range(self.num_epochs)):
            self._train_epoch(X, y)

    def predict(self, X) -> torch.Tensor:
        """Method outputs prediction of the network. Needed to comply with sklearn interface"""
        X = self._adjust_type_dims_device(x=X, target_dim=3)

        return self.forward(token=X)