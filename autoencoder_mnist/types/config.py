from datetime import datetime

from pydantic import BaseModel

from .. import config as Defaults


class Config(BaseModel):
    model: str # "cnn" | "fc" | "pca"
    experiment_name: str
    n_components: int

    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 10
    data_dir: str = str(Defaults.RAW_DATA_DIR)
    use_gpu: bool = False
