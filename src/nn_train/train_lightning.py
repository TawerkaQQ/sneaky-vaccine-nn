import os
import warnings
from typing import Any

import cv2
import lightning as pl
import mlflow
import numpy as np
import torch
import torch.nn as nn
from dotenv import find_dotenv, load_dotenv
from dynaconf import Dynaconf
from lightning.pytorch.loggers import MLFlowLogger

from src.nn_train.models.simple_cnn_model import SimpleLandmarkNet as model
from src.nn_train.datasets.dataset import CustomDataset
load_dotenv(find_dotenv())
config = Dynaconf(settings_file=["train_config.yaml"])

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_uri)

mlflow_logger = MLFlowLogger(
    experiment_name=config["MLFLOW_LOGGING_EXPERIMENT_NAME"],
    tracking_uri=mlflow_tracking_uri,
    log_model="all",
)

torch.set_float32_matmul_precision("high")

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model()
        self.learning_rate = config["LEARNING_RATE"]
        self.loss = nn.MSELoss()
        self.epoch = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        """Train step logic"""
        data, target = batch
        output = self.forward(data)
        loss = self.loss(output, target)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> list[Any]:
        """optim settings"""
        optimizer = torch.optim.RAdam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config["WEIGHT_DECAY"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config["MILESTONES"], gamma=config["GAMMA"]
        )

        return [optimizer], [scheduler]


if __name__ == "__main__":

    """TRAIN MODEL"""

    if config["PRETRAINED"]:
        weights = config["WEIGHTS"]
        model = LitModel.load_from_checkpoint(weights)
    else:
        model = LitModel()

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        filename="best_model",
        monitor="train_loss",
        mode="min",
    )

    checkpoint_callback.CHECKPOINT_JOIN_CHAR = "+"
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    with warnings.catch_warnings(action="ignore"):
        trainer = pl.Trainer(
            max_epochs=config["EPOCHS"],
            devices=1 if not config["DEVICE"] else config["DEVICE"],
            accelerator=config["ACCELERATOR"],
            logger=mlflow_logger if config["MLFLOW_LOGGING_ENABLED"] else None,
            enable_progress_bar=True,
            enable_model_summary=True,
            accumulate_grad_batches=32,
            callbacks=[checkpoint_callback],
            strategy="ddp_find_unused_parameters_true",
        )

    dataset = CustomDataset(config["TRAIN_DATASET"])
    trainer.fit(model, dataset)

    sorted_models = sorted(
        os.listdir("./checkpoints"), key=lambda x: (extract_version(x), x)
    )
    if config["CONVERT_MODEL"]:
        convert_model_to_torchscript(sorted_models[-1], f"./checkpoints/{sorted_models[-1]}")

