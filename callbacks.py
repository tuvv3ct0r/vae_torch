import os
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning import Callback, Trainer, LightningModule


class LossPlotCallback(Callback):
    def __init__(self, plot_frequency: int = 10):
        super().__init__()
        self.plot_frequency = plot_frequency
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        train_loss = trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        val_loss = trainer.callback_metrics.get('val_loss_epoch')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

        # Plot only if we have at least one loss and it's time to plot
        if (trainer.current_epoch + 1) % self.plot_frequency == 0 and self.train_losses:
            self.plot_losses(trainer.logger.log_dir, trainer.current_epoch + 1)

    def plot_losses(self, log_dir: str, epoch: int):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', marker='o')

        # Plot validation losses only if available and lengths match
        if self.val_losses and len(self.val_losses) >= len(self.train_losses):
            plt.plot(epochs, self.val_losses[:len(self.train_losses)], label='Validation Loss', marker='s')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        graph_dir = os.path.join(log_dir, 'graph')
        Path(graph_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(os.path.join(graph_dir, f'loss_plot_epoch_{epoch}.png'))
        plt.close()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.train_losses:
            self.plot_losses(trainer.logger.log_dir, trainer.current_epoch)