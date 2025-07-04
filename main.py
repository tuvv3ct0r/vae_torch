import os
import yaml
import argparse
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from dataset import CIFAR10DataModule
from model.vae import VAE
from train import VAETrainer
from callbacks import LossPlotCallback


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="vae_torch/configs/vae.yaml", help="Path to config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name="vae_logs")

seed_everything(config['exp_params']['manual_seed'], True)

model = VAE(**config['model_params'], kld_weight=config['exp_params']['kld_weight'])
trainer = VAETrainer(model, config['exp_params'])
data = CIFAR10DataModule(
    **config['data_params'],
    pin_memory=len(config['trainer_params']['devices']) > 0 and config['trainer_params']['accelerator'] == "gpu"
)

data.setup()

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), 
                                     monitor="val_loss",
                                     save_last=True),
                     LossPlotCallback(plot_frequency=10),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']} =======")
runner.fit(trainer, datamodule=data)