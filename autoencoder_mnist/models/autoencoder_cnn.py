import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderCNN(pl.LightningModule):
    # Required for PyTorch Lightning model description
    example_input_array = torch.zeros((1, 1, 28, 28))

    def __init__(self, n_components: int, lr: float):
        super().__init__()
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.Flatten(),  # b, 32
            nn.Linear(32, n_components),  # b, n_components
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 32),  # b, 32
            nn.Unflatten(1, (8, 2, 2)),  # b, 8, 2, 2
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("tr_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
