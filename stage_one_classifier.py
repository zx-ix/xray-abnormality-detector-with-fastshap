import logging
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier_utilities import set_metrics, set_schedule, compute_metrics, epoch_wrapup
from collections import defaultdict

class StageOneClassifier(pl.LightningModule):
    def __init__(
        self,
        download_weight: bool, 
        load_path: str or None,
        target_type: str,
        output_dim: int,
        checkpoint_metric: str or None,
        optim_type: str or None,
        learning_rate: float or None,
        weight_decay: float or None,
        decay_power: str or None,
        warmup_steps: int or None,
        batch_size: int,
        distillation: bool = False,
        distillation_temperature: float = 1.0,
        distillation_alpha: float = 0.5,
        loss_weight: list[float] | None = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.logger_ = logging.getLogger(__name__)
        assert not (self.hparams.download_weight and self.hparams.load_path is not None), \
            "'download_weight' and 'load_path' cannot be activated at the same time as the downloaded weight will be overwritten by weights in 'load_path'."

        if self.hparams.distillation:
            self.backbone = torch.hub.load(
                'facebookresearch/deit:main', 
                'deit_base_distilled_patch16_224', 
                pretrained=self.hparams.download_weight
            )
        else:
            self.backbone = torch.hub.load(
                'facebookresearch/deit:main', 
                'deit_base_patch16_224', 
                pretrained=self.hparams.download_weight
            )

        if self.hparams.download_weight:
            self.logger_.info("Backbone initialized with downloaded pretrained weights.")
        else:
            self.logger_.info("Backbone randomly initialized.")

        self.backbone.head = nn.Linear(self.backbone.embed_dim, self.hparams.output_dim, bias=True)

        if self.hparams.distillation:
            self.backbone.head_dist = nn.Linear(self.backbone.embed_dim, self.hparams.output_dim, bias=True)
            
        if self.hparams.load_path is not None:
            checkpoint = torch.load(self.hparams.load_path, map_location=self.device)
            state_dict = checkpoint["state_dict"]
            ret = self.load_state_dict(state_dict, strict=False)
            self.logger_.info(f"Model parameters updated from checkpoint {self.hparams.load_path}")
            self.logger_.info(f"Missing keys: {ret.missing_keys}")
            self.logger_.info(f"Unexpected keys: {ret.unexpected_keys}")
        
        for param in self.backbone.parameters():
            param.requires_grad = True

        set_metrics(self)

    def configure_optimizers(self):
        return set_schedule(self)

    def forward(self, images):
        x = self.backbone.forward_features(images)
        if self.hparams.distillation:
            # Expect x to be a tuple: (class_token_features, distillation_token_features)
            if isinstance(x, tuple):
                x_cls, x_dist = x
                logits_cls = self.backbone.head(x_cls)
                logits_dist = self.backbone.head_dist(x_dist)
                return logits_cls, logits_dist
            else:
                return self.backbone.head(x)
        else:
            return self.backbone.head(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["body_part_idx"]
        # convert one-hot encoded targets to class indices
        targets_int = torch.argmax(targets, dim=1)

        if self.hparams.distillation:
            logits_cls, logits_dist = self(images)
            loss_cls = F.cross_entropy(logits_cls, targets_int)
            # simulate teacher predictions with detached class token outputs (for future work)
            teacher_preds = logits.detach()
            T = self.hparams.distillation_temperature
            loss_dist = F.kl_div(
                F.log_softmax(logits_dist/T, dim=1),
                F.softmax(teacher_preds/T, dim=1),
                reduction='batchmean'
            )*(T*T)

            loss = self.hparams.distillation_alpha*loss_cls+(1-self.hparams.distillation_alpha)*loss_dist
        else:
            logits = self(images)
            loss = F.cross_entropy(logits, targets_int)
        
        _ = compute_metrics(self, logits, targets_int, phase='train')
        return loss

    def on_train_epoch_end(self):
        epoch_wrapup(self, phase='train')
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["body_part_idx"]
        targets_int = torch.argmax(targets, dim=1)
        if self.hparams.distillation:
            logits_cls, _ = self(images)
            _ = compute_metrics(self, logits_cls, targets_int, phase='val')
        else:
            logits = self(images)
            _ = compute_metrics(self, logits, targets_int, phase='val')

    def on_validation_epoch_end(self):
        epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["body_part_idx"]
        targets_int = torch.argmax(targets, dim=1)
        if self.hparams.distillation:
            logits_cls, _ = self(images)
            _ = compute_metrics(self, logits_cls, targets_int, phase='test')
        else:
            logits = self(images)
            _ = compute_metrics(self, logits, targets_int, phase='test')

    def on_test_epoch_end(self):
        epoch_wrapup(self, phase='test')