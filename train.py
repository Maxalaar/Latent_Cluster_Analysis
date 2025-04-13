from datetime import timedelta
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

from torchvision.datasets import MNIST

from architecture.autoencoder import Autoencoder
from utilities.data_module import DataModule
from utilities.training_configuration import TrainingConfiguration

dataset_directory='./dataset'
tensor_board_path = './tensor_board'
save_checkpoint_path = './experiments'

if __name__ == '__main__':
    configuration = TrainingConfiguration(
        experiment_name='autoencoder_10_minutes',
        dataset_class=MNIST,  # MNIST, CIFAR10
        model_class=Autoencoder,
        maximum_training_time=timedelta(minutes=10),
        checkpoint_interval_time=timedelta(minutes=2),
        number_model_to_train=6
    )

    # configuration = TrainingConfiguration(
    #     experiment_name='debug',
    #     dataset_class=MNIST,
    #     model_class=Autoencoder,
    #     maximum_training_time=timedelta(minutes=0.5),
    #     checkpoint_interval_time=timedelta(minutes=0.25),
    #     number_model_to_train=6,
    # )

    data_module = DataModule(
        dataset_class=configuration.dataset_class,
        dataset_directory=dataset_directory,
        batch_size=configuration.batch_size,
        train_validation_split=configuration.train_validation_split,
    )
    data_module.prepare_data()
    data_module.setup()

    for i in range(configuration.number_model_to_train):
        print(f"Starting training model {i + 1}/{configuration.number_model_to_train}")

        early_stopping = EarlyStopping(
            monitor='validation_loss',
            min_delta=0.00,
            patience=configuration.patience,
            verbose=True,
            mode='min',
        )

        model = configuration.model_class(
            input_shape=data_module.shape,
            output_shape=data_module.shape,
        )

        logger = TensorBoardLogger(
            save_dir=save_checkpoint_path,
            name=configuration.experiment_name + '/model/',
        )
        configuration.save(Path(logger.log_dir))

        checkpoint_callback = ModelCheckpoint(
            train_time_interval=configuration.checkpoint_interval_time,
            save_top_k=1,
        )

        trainer = pl.Trainer(
            max_epochs=-1,
            max_time=configuration.maximum_training_time,
            logger=logger,
            check_val_every_n_epoch=configuration.validation_interval,
            callbacks=[checkpoint_callback, early_stopping],
        )

        trainer.fit(
            model=model,
            datamodule=data_module,
        )