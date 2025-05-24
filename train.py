import os
import torch
from torch import optim
from pytorch_lightning import LightningModule
import torchvision.utils as vutils


class VAETrainer(LightningModule):
    def __init__(self, model, params: dict) -> None:
        super().__init__()
        self.model = model
        self.params = params
        self.save_hyperparameters(params)
        if 'retain_first_backpass' in self.params:
            self.hold_graph = self.params['retain_first_backpass']
        else:
            self.hold_graph = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        results = self.forward(real_images)
        train_loss = self.model.loss_function(*results, batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        real_img, _ = batch
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results, batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        return val_loss['loss']
    
    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)

        # Encode input to get latent vector z
        mu, log_var = self.model.encoder(test_input)
        z = self.model.reparameterize(mu, log_var)
        recons = self.model.generate(z)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir, 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir, 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Exception as e:
            self.log("sample_error", str(e))
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        if 'scheduler_gamma' in self.params:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=self.params['scheduler_gamma'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer