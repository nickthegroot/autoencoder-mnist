from enum import Enum

from pydantic import BaseModel

from .. import consts as Consts


class ModelType(str, Enum):
    AUTOENCODER_FC = "fc"
    AUTOENCODER_CNN = "cnn"
    PCA = "pca"


class Config(BaseModel):
    model: ModelType
    n_components: int
    data_dir: str = str(Consts.RAW_DATA_DIR)

    # Auto-filled in train
    experiment_name: str

    # Neural Nets Only
    learning_rate: float = 0.001
    batch_size: int = 1024
    max_epochs: int = 10
    use_gpu: bool = False
    num_workers: int = 1
