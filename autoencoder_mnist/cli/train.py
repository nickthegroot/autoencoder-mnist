import json
from pathlib import Path

import click
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ..data.mnist_loader import MNISTLoader
from ..models.autoencoder_cnn import AutoencoderCNN
from ..models.autoencoder_fc import AutoencoderFC
from ..types.config import Config
from .. import config as Consts

def train_kmeans_nn(model: pl.LightningModule, datamodule: pl.LightningDataModule, config: Config):
    # Train Model
    model_dir = Consts.MODEL_DIR / config.experiment_name
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=str(model_dir),
        accelerator="gpu" if config.use_gpu else "cpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(model, datamodule=datamodule)

    # trainer.save_checkpoint(str(model_dir / "model.ckpt"), weights_only = True)

    enc = model.encoder
    enc.eval()

    # Train KMeans
    with torch.no_grad():
        kmeans = MiniBatchKMeans(n_clusters = 10)
        for X, _ in datamodule.train_dataloader():
            pred = enc(X)
            kmeans.partial_fit(pred.flatten(1).numpy())
        return enc, kmeans

def train_kmeans_pca(datamodule: pl.LightningDataModule, config: Config):
    datamodule.prepare_data()
    datamodule.setup('fit')

    # Train PCA
    if config.model == "pca":
        # NOTE: Should use high batch size, as only one "batch" is used
        pca = PCA(n_components=config.n_components)
        X = next(iter(datamodule.train_dataloader()))[0].view(config.batch_size, -1).numpy()
        pca.fit(X)
    elif config.model == "inc_pca":
        pca = IncrementalPCA(n_components = config.n_components)
        for X, _ in tqdm(datamodule.train_dataloader()):
            X = X.view(config.batch_size, -1).numpy()
            pca.partial_fit(X)

    # Train KMeans
    kmeans = MiniBatchKMeans(n_clusters = 10)
    for X, _ in datamodule.train_dataloader():
        pred = pca.transform(X.view(X.shape[0], -1).numpy())
        kmeans.partial_fit(pred)
    return pca, kmeans

@click.command()
@click.option("--config-path", type=click.Path(exists=True))
def run(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        config = Config(**config, experiment_name=Path(config_path).parent.stem)
        f.close()

    datamodule = MNISTLoader(config.data_dir, config.batch_size)
    
    if config.model == "cnn":
        enc, kmeans = train_kmeans_nn(
            AutoencoderCNN(config.n_components, config.learning_rate),
            datamodule,
            config
        )
    elif config.model == "fc":
        enc, kmeans = train_kmeans_nn(
            AutoencoderFC(config.n_components, config.learning_rate),
            datamodule,
            config
        )
    elif config.model == "pca":
        enc, kmeans = train_kmeans_pca(datamodule, config)
    elif config.model == "inc_pca":
        enc, kmeans = train_kmeans_pca(datamodule, config)

    print(kmeans.cluster_centers_)
