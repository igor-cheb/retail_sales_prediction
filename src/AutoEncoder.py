import torch

class AutoEncoder (torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_epochs: int=100):
        super().__init__()
        self.num_epochs = num_epochs

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, latent_dim),
            torch.nn.LeakyReLU()
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, input_dim)
            )

        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, X):
        encoder_pred = self.encoder(X)
        decoder_pred = self.decoder(encoder_pred)
        return decoder_pred
        
    def fit(self, X, y):
        for _ in range(self.num_epochs):
            pred = self.forward(X)
            print(pred.shape)
            loss = torch.nn.functional.mse_loss(input=pred, target=y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


    def predict(self, X):
        """To comply with sklearn interface"""
        return self.encoder(X)