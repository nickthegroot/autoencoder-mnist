import json

import click
import pytorch_lightning as pl

from ..data.mnist_loader import MNISTLoader
from ..models.autoencoder_cnn import AutoencoderCNN
from ..models.autoencoder_fc import AutoencoderFC
from ..types.config import Config


@click.command()
@click.Option("--config-path", type=click.Path(exists=True))
def train(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        config = Config(**config)
        f.close()

    datamodule = MNISTLoader(config.data_dir, config.batch_size)

    if config.model == "cnn":
        model = AutoencoderCNN(config.learning_rate)
    elif config.model == "fc":
        model = AutoencoderFC(config.learning_rate)
    # TODO: implement PCA

    experiment_path = Config.CHECKPOINT_DIR / config.experiment_name
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        default_root_dir=str(experiment_path),
        accelerator="gpu" if Config.use_gpu else "cpu",
    )
    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint(str(experiment_path / "model.ckpt"), weights_only = True)
