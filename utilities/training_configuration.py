from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Type
import yaml

from architecture.autoencoder import Autoencoder


@dataclass
class TrainingConfiguration:
    experiment_name: str = None
    dataset_class: Type = None
    model_class: Type = None
    model_configuration: dict = field(default_factory=dict)
    patience: int = float('inf')
    checkpoint_interval_time: timedelta = timedelta(minutes=1)
    validation_interval: int = 20
    maximum_training_time: timedelta = timedelta(minutes=30)
    number_model_to_train: int = 10
    batch_size: int = 64
    train_validation_split: Tuple[float, float] = (0.9, 0.1)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / 'training_configuration.yaml'
        configuration_dictionary = {key: value for key, value in self.__dict__.items()}

        with open(file_path, 'w') as file:
            yaml.dump(configuration_dictionary, file)

    def load(self, yaml_path: Path):
        if not yaml_path.exists():
            raise FileNotFoundError(f"The file {yaml_path} does not exist.")

        with yaml_path.open('r') as file:
            dictionary = yaml.load(file, Loader=yaml.UnsafeLoader)

        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

        return self