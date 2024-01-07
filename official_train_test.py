import os
import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics.functional.regression import mean_absolute_error
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        val_mae = mean_absolute_error(x_hat, x)
        self.log("val_loss", val_loss)
        self.log("val_MAE", val_mae)


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

transform = transforms.ToTensor()

train_set = datasets.MNIST(root="./", download=True, train=True, transform=transform)
# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
train_loader = DataLoader(train_set, batch_size=64)
valid_loader = DataLoader(valid_set, batch_size=64)
test_set = datasets.MNIST(root="./", download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=64)


# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=200, max_epochs=2, callbacks=[EarlyStopping(monitor="val_MAE", mode="min")])
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(model=autoencoder, dataloaders=test_loader)
