# References:
# Adapted from ViTShapley:
# I. Covert, C. Kim, and S.-I. Lee, "Learning to estimate Shapley values with vision transformers,"
# presented at the International Conference on Learning Representations 2023, Kigali, Rwanda, 2023.

import logging
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, Accuracy, Precision, Recall, F1Score, CohenKappa, AUROC, ROC
from transformers import get_cosine_schedule_with_warmup, AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def set_schedule(pl_module):
    if pl_module.hparams.optim_type is None:
        return [None], [None]

    if pl_module.hparams.optim_type == "AdamW":
        optimizer = AdamW(
            params=pl_module.parameters(),
            lr=pl_module.hparams.learning_rate,
            weight_decay=pl_module.hparams.weight_decay
        )
    elif pl_module.hparams.optim_type == "Adam":
        optimizer = torch.optim.Adam(
            pl_module.parameters(),
            lr=pl_module.hparams.learning_rate,
            weight_decay=pl_module.hparams.weight_decay
        )
    elif pl_module.hparams.optim_type == "SGD":
        optimizer = torch.optim.SGD(
            pl_module.parameters(),
            lr=pl_module.hparams.learning_rate,
            momentum=0.9,
            weight_decay=pl_module.hparams.weight_decay
        )
    else:
        raise NotImplementedError("Unsupported optimizer type.")

    trainer = pl_module.trainer
    if trainer.max_steps is None or trainer.max_steps == -1:
        # total steps = (batches per epoch) * epochs / accumulate_grad_batches
        steps_per_epoch = len(trainer.datamodule.train_dataloader())
        max_steps = (steps_per_epoch * trainer.max_epochs) // trainer.accumulate_grad_batches
    else:
        max_steps = trainer.max_steps

    if pl_module.hparams.decay_power == "cosine":
        scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=pl_module.hparams.warmup_steps,
                num_training_steps=max_steps
            ),
            "interval": "step"
        }
    else:
        raise NotImplementedError("Only cosine scheduler is implemented.")

    return [optimizer], [scheduler]

def plot_roc_curves(fpr, tpr, phase, target_type="multiclass", class_names=None, filename_prefix="roc_curve"):
    plt.figure()

    if target_type == "binary":
        # Handle a single FPR/TPR array
        if class_names is None:
            class_names = ["Negative", "Positive"]

        roc_auc = auc(fpr.cpu(), tpr.cpu())
        plt.plot(fpr.cpu(), tpr.cpu(), label=f"ROC (area = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"Stage two ROC curve: ({phase.capitalize()})")
        plt.legend(loc="lower right")
        plt.savefig(f"{filename_prefix}_{phase}_binary.png")
        plt.close()

    elif target_type == "multiclass":
        # fpr[i], tpr[i] for each class
        if class_names is None:
            class_names = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]

        num_classes = len(class_names)
        for i in range(num_classes):
            roc_auc = auc(fpr[i].cpu(), tpr[i].cpu())
            plt.plot(fpr[i].cpu(), tpr[i].cpu(),
                     label=f"{class_names[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"Stage one ROC curve: ({phase.capitalize()})")
        plt.legend(loc="lower right")
        plt.savefig(f"{filename_prefix}_{phase}_multiclass.png")
        plt.close()
    else:
        raise ValueError("target_type must be 'binary' or 'multiclass'.")

def set_metrics(pl_module):
    for phase in ["train", "val", "test"]:
        if pl_module.hparams.target_type == "binary":
            setattr(pl_module, f"{phase}_loss", MeanMetric())
            setattr(pl_module, f"{phase}_accuracy", Accuracy(task="binary"))
            setattr(pl_module, f"{phase}_precision", Precision(task="binary"))
            setattr(pl_module, f"{phase}_recall", Recall(task="binary"))
            setattr(pl_module, f"{phase}_f1", F1Score(task="binary"))
            setattr(pl_module, f"{phase}_cohenkappa", CohenKappa(task="binary", weights="quadratic"))
            setattr(pl_module, f"{phase}_auroc", AUROC(task="binary"))
            setattr(pl_module, f"{phase}_roc", ROC(task="binary"))

        elif pl_module.hparams.target_type == "multiclass":
            num_classes = pl_module.hparams.output_dim
            setattr(pl_module, f"{phase}_loss", MeanMetric())
            setattr(pl_module, f"{phase}_accuracy", Accuracy(task="multiclass", num_classes=num_classes, average="macro"))
            setattr(pl_module, f"{phase}_precision", Precision(task="multiclass", num_classes=num_classes, average="macro"))
            setattr(pl_module, f"{phase}_recall", Recall(task="multiclass", num_classes=num_classes,average="macro"))
            setattr(pl_module, f"{phase}_f1", F1Score(task="multiclass", num_classes=num_classes, average="macro"))
            setattr(pl_module, f"{phase}_cohenkappa", CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic"))
            setattr(pl_module, f"{phase}_auroc", AUROC(task="multiclass", num_classes=num_classes, average="macro"))
            setattr(pl_module, f"{phase}_roc", ROC(task="multiclass", num_classes=num_classes))
        else:
            raise NotImplementedError("target_type should be 'binary' or 'multiclass'.")

def log_and_reset_metric(pl_module, phase, metric_name, prog_bar=True):
    metric_obj = getattr(pl_module, f"{phase}_{metric_name}")
    val = metric_obj.compute()
    metric_obj.reset()
    pl_module.log(f"{phase}/epoch_{metric_name}", val, prog_bar=prog_bar,
                  on_step=False, on_epoch=True)
    return val

def epoch_wrapup(pl_module, phase):
    if pl_module.hparams.target_type == "binary":
        loss = log_and_reset_metric(pl_module, phase, "loss")
        accuracy = log_and_reset_metric(pl_module, phase, "accuracy")
        precision = log_and_reset_metric(pl_module, phase, "precision")
        recall = log_and_reset_metric(pl_module, phase, "recall")
        f1 = log_and_reset_metric(pl_module, phase, "f1")
        cohenkappa = log_and_reset_metric(pl_module, phase, "cohenkappa")
        auroc = log_and_reset_metric(pl_module, phase, "auroc")

        fpr, tpr, thresholds = getattr(pl_module, f"{phase}_roc").compute()
        getattr(pl_module, f"{phase}_roc").reset()
        filename = f"{pl_module.hparams.body_part_filter}_roc_curve"
        plot_roc_curves(fpr, tpr, phase=phase, target_type="binary", filename_prefix=filename)

        if pl_module.hparams.checkpoint_metric == "CohenKappa":
            checkpoint_metric = cohenkappa
        elif pl_module.hparams.checkpoint_metric == "AUC":
            checkpoint_metric = auroc
        elif pl_module.hparams.checkpoint_metric == "accuracy":
            checkpoint_metric = accuracy
        else:
            raise NotImplementedError("Not supported checkpoint metric")

        pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)

    elif pl_module.hparams.target_type == "multiclass":
        loss = log_and_reset_metric(pl_module, phase, "loss")
        accuracy = log_and_reset_metric(pl_module, phase, "accuracy")
        precision = log_and_reset_metric(pl_module, phase, "precision")
        recall = log_and_reset_metric(pl_module, phase, "recall")
        f1 = log_and_reset_metric(pl_module, phase, "f1")
        cohenkappa = log_and_reset_metric(pl_module, phase, "cohenkappa")
        auroc = log_and_reset_metric(pl_module, phase, "auroc")

        fpr, tpr, thresholds = getattr(pl_module, f"{phase}_roc").compute()
        getattr(pl_module, f"{phase}_roc").reset()
        plot_roc_curves(fpr, tpr, phase=phase, target_type="multiclass")

        if pl_module.hparams.checkpoint_metric == "CohenKappa":
            checkpoint_metric = cohenkappa
        elif pl_module.hparams.checkpoint_metric == "AUC":
            checkpoint_metric = auroc
        elif pl_module.hparams.checkpoint_metric == "accuracy":
            checkpoint_metric = accuracy
        else:
            raise NotImplementedError("Not supported checkpoint metric")

        pl_module.log(f"{phase}/checkpoint_metric", checkpoint_metric)
    else:
        raise NotImplementedError("target_type should be 'binary' or 'multiclass'.")

def compute_metrics(pl_module, logits, labels, phase):
    target_type = pl_module.hparams.target_type
    loss_weight = getattr(pl_module.hparams, "loss_weight", None)

    if target_type == "binary":
        if loss_weight is not None:
            pos_weight_tensor = torch.tensor(loss_weight, device=logits.device, dtype=logits.dtype)
            loss = pos_weight_tensor[0]/(pos_weight_tensor[0] + pos_weight_tensor[1])*F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=torch.tensor(pos_weight_tensor[1] / pos_weight_tensor[0]).float()
)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        prob = torch.sigmoid(logits)
        labels = labels.long()
        
        acc = getattr(pl_module, f"{phase}_accuracy")(prob, labels)
        pl_module.log(f"{phase}/accuracy", acc, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        recall = getattr(pl_module, f"{phase}_recall")(prob, labels)
        pl_module.log(f"{phase}/recall", recall, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        precision = getattr(pl_module, f"{phase}_precision")(prob, labels)
        pl_module.log(f"{phase}/precision", precision, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        f1 = getattr(pl_module, f"{phase}_f1")(prob, labels)
        pl_module.log(f"{phase}/f1", f1, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa")(prob, labels)
        pl_module.log(f"{phase}/cohenkappa", cohenkappa, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        auroc = getattr(pl_module, f"{phase}_auroc")(prob, labels)
        pl_module.log(f"{phase}/auroc", auroc, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        _ = getattr(pl_module, f"{phase}_roc").update(prob, labels)
        
        return loss

    elif target_type == "multiclass":
        if loss_weight is not None:
            weight_tensor = torch.tensor(loss_weight, device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, labels, weight=weight_tensor)
        else:
            loss = F.cross_entropy(logits, labels)

        loss = getattr(pl_module, f"{phase}_loss")(loss)
        pl_module.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        prob = torch.softmax(logits, dim=1)
        
        acc = getattr(pl_module, f"{phase}_accuracy")(prob, labels)
        pl_module.log(f"{phase}/accuracy", acc, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        rec = getattr(pl_module, f"{phase}_recall")(prob, labels)
        pl_module.log(f"{phase}/recall", rec, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        prec = getattr(pl_module, f"{phase}_precision")(prob, labels)
        pl_module.log(f"{phase}/precision", prec, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        f1 = getattr(pl_module, f"{phase}_f1")(prob, labels)
        pl_module.log(f"{phase}/f1", f1, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        cohenkappa = getattr(pl_module, f"{phase}_cohenkappa")(prob, labels)
        pl_module.log(f"{phase}/cohenkappa", cohenkappa, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        auroc = getattr(pl_module, f"{phase}_auroc")(prob, labels)
        pl_module.log(f"{phase}/auroc", auroc, on_step=False, on_epoch=True, batch_size=pl_module.hparams.batch_size)

        _ = getattr(pl_module, f"{phase}_roc").update(prob, labels)

        return loss

    else:
        raise NotImplementedError("target_type should be 'binary' or 'multiclass'.")