import pandas as pd
import numpy as np
import torch

class MLPModel(torch.nn.Module):
    def __init__(self, input_size: int, 
                 epochs: int=3, 
                 batch_size: int=512, 
                 verbose: bool=False):
        super().__init__()
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
            )
        self.optim = torch.optim.Adam(self.parameters())
        
    def predict(self, X: np.ndarray):
        loc_X = torch.tensor(X, dtype=torch.float32) #
        return self.layers(loc_X).detach().numpy()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        loc_X = torch.tensor(X, dtype=torch.float32)
        loc_y = torch.tensor(y, dtype=torch.float32)
        
        for _ in range(self.epochs):
            if self.verbose: print(f'epoch: {_}')
            for i in range(0, X.shape[0], self.batch_size):
                ixs = np.array(range(i, min([X.shape[0], i + self.batch_size])))
                batch_X = loc_X[ixs]
                batch_y = loc_y[ixs]
                pred = self.layers(batch_X)
                loss = torch.nn.functional.l1_loss(input=pred.flatten(), 
                                                   target=batch_y.flatten())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()