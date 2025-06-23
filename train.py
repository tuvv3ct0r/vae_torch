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
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kld_losses = []
        self.val_recon_losses = []
        self.val_kld_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        results = self.forward(real_images)
        train_loss = self.model.loss_function(*results, batch_idx=batch_idx)
        if torch.isnan(train_loss['loss']):
            print(f"NaN detected in training loss at batch {batch_idx}")
        self.train_losses.append(train_loss['loss'].detach())
        self.train_recon_losses.append(train_loss['Reconstruction_Loss'].detach())
        self.train_kld_losses.append(train_loss['KLD'].detach())
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    
    def on_train_epoch_end(self):
        if self.train_losses:
            avg_loss = torch.stack([x for x in self.train_losses if not torch.isnan(x)]).mean() if self.train_losses else torch.tensor(0.0)
            avg_recon = torch.stack([x for x in self.train_recon_losses if not torch.isnan(x)]).mean() if self.train_recon_losses else torch.tensor(0.0)
            avg_kld = torch.stack([x for x in self.train_kld_losses if not torch.isnan(x)]).mean() if self.train_kld_losses else torch.tensor(0.0)
            self.log('train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train_recon_loss_epoch', avg_recon, on_epoch=True, sync_dist=True)
            self.log('train_kld_loss_epoch', avg_kld, on_epoch=True, sync_dist=True)
            self.train_losses = []
            self.train_recon_losses = []
            self.train_kld_losses = []
    
    def validation_step(self, batch, batch_idx):
        real_img, _ = batch
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results, batch_idx=batch_idx)
        if torch.isnan(val_loss['loss']):
            print(f"NaN detected in validation loss at batch {batch_idx}")
        self.val_losses.append(val_loss['loss'].detach())
        self.val_recon_losses.append(val_loss['Reconstruction_Loss'].detach())
        self.val_kld_losses.append(val_loss['KLD'].detach())
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        return val_loss['loss']
    
    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = torch.stack([x for x in self.val_losses if not torch.isnan(x)]).mean() if self.val_losses else torch.tensor(0.0)
            avg_recon = torch.stack([x for x in self.val_recon_losses if not torch.isnan(x)]).mean() if self.val_recon_losses else torch.tensor(0.0)
            avg_kld = torch.stack([x for x in self.val_kld_losses if not torch.isnan(x)]).mean() if self.val_kld_losses else torch.tensor(0.0)
            self.log('val_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_recon_loss_epoch', avg_recon, on_epoch=True, sync_dist=True)
            self.log('val_kld_loss_epoch', avg_kld, on_epoch=True, sync_dist=True)
            self.val_losses = []
            self.val_recon_losses = []
            self.val_kld_losses = []
    
    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)

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
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "gradient_clip_val": 1.0,
                "gradient_clip_algorithm": "norm"
            }
        return optimizer