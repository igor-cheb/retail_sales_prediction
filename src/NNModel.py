import numpy as np
import torch

class NNModel(torch.nn.Module):
    def __init__(self, input_size: int, epochs=1, batch_size=1024):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
            )
        self.optim = torch.optim.Adam(self.parameters())
        
    def predict(self, X: np.ndarray):
        loc_X = torch.tensor(X, dtype=torch.float32) #
        return self.layers(loc_X).detach().numpy()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        loc_X = torch.tensor(X, dtype=torch.float32)
        loc_y = torch.tensor(y, dtype=torch.float32)
        
        for _ in range(self.epochs):
            print(f'epoch: {_}')
            for i in range(0, X.shape[0], self.batch_size):
                ixs = range(i, min([X.shape[0], i + self.batch_size]))
                batch_X = loc_X[ixs]
                batch_y = loc_y[ixs]
                pred = self.layers(batch_X)
                loss = torch.nn.functional.l1_loss(input=pred, 
                                                   target=batch_y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()