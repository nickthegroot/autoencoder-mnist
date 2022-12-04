from datetime import datetime

from pydantic import BaseModel

from .. import config as Defaults


class Config(BaseModel):
    model: str # "cnn" | "fc" | "pca"

    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    data_dir: str = str(Defaults.RAW_DATA_DIR)
    experiment_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    use_gpu: bool = False
