import json
from pathlib import Path

import click
import joblib
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .. import consts as Consts
from ..data.mnist_loader import MNISTLoader
from ..models.autoencoder_cnn import AutoencoderCNN
from ..models.autoencoder_fc import AutoencoderFC
from ..models.pca import train_pca
from ..types.config import Config, ModelType


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        config = Config(**config, experiment_name=Path(config_path).parent.stem)
        f.close()

    if config.model == ModelType.AUTOENCODER_FC:
        train_nn(AutoencoderFC(config.n_components, config.learning_rate), config)
    elif config.model == ModelType.AUTOENCODER_CNN:
        train_nn(AutoencoderCNN(config.n_components, config.learning_rate), config)
    elif config.model == ModelType.PCA:
        pca = train_pca(config.data_dir, config.n_components)
        joblib.dump(pca, Consts.MODEL_DIR / config.experiment_name / "model.pkl")


def train_nn(model: pl.LightningModule, config: Config):
    # Train Model
    datamodule = MNISTLoader(config.data_dir, config.batch_size, config.num_workers)

    model_dir = Consts.MODEL_DIR / config.experiment_name
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=str(model_dir),
        accelerator="gpu" if config.use_gpu else "cpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(str(model_dir / "model.ckpt"), weights_only=True)
