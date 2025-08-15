import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import torch
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from MURA_datamodule import MURADataModule
from stage_one_classifier import StageOneClassifier
from stage_two_classifier import StageTwoClassifier


def main():
    # Configuration
    DATA_DIR = "/home/fypstudent/Documents/zixupuah-fyp/MURA-v1.1"
    assert os.path.isdir(DATA_DIR), f"Dataset directory {DATA_DIR} not found"

    BATCH_SIZE = 64
    NUM_WORKERS = 4
    TOP_K = 5

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # DataModule setup
    datamodule = MURADataModule(
        dataset_location=DATA_DIR,
        transforms_original=transform,
        transforms_augment=transform,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        body_part_filter=None,
        test_data_split="test",
    )
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_dataset
    test_loader = datamodule.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Stage 1 model
    stage1_ckpt = "./VS_code_stage_one_logs/lightning_logs/version_0/checkpoints/best-checkpoint.ckpt"
    stage1 = StageOneClassifier.load_from_checkpoint(stage1_ckpt, map_location=device)
    stage1.to(device).eval()

    # Body parts list
    BODY_PARTS = test_dataset.body_part_strs

    # Load Stage 2 models
    stage2_models = {}
    for bp in BODY_PARTS:
        ckpt_path = f"./VS_code_stage_two_logs/{bp}/lightning_logs/version_0/checkpoints/{bp}_best-checkpoint.ckpt"
        model2 = StageTwoClassifier.load_from_checkpoint(ckpt_path, map_location=device)
        model2.to(device).eval()
        stage2_models[bp] = model2

    # Binary confusion matrix metric
    confmat = ConfusionMatrix(task="binary", num_classes=2).to(device)

    # Storage for examples
    results = {bp: {"correct": [], "incorrect": []} for bp in BODY_PARTS}

    # Denormalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Inference
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            true_bins = batch["label"].to(device)
            true_parts = batch["body_part_idx"].to(device).argmax(dim=1)

            # Stage 1: predict body part
            logits1 = stage1(images)
            pred_parts = logits1.argmax(dim=1)

            for i in range(images.size(0)):
                img_tensor = images[i].unsqueeze(0)
                bp_true = BODY_PARTS[true_parts[i].item()]
                bp_pred = BODY_PARTS[pred_parts[i].item()]

                # Stage 2
                log2 = stage2_models[bp_pred](img_tensor)
                prob_abnormal = torch.sigmoid(log2).item()
                # Confidence of the predicted class
                pred_bin = int(prob_abnormal > 0.5)
                conf = prob_abnormal if pred_bin == 1 else (1 - prob_abnormal)
                true_bin = int(true_bins[i].item())

                # Update confusion matrix
                confmat.update(
                    torch.tensor([pred_bin], device=device),
                    torch.tensor([true_bin], device=device)
                )

                # Record example
                entry = {
                    "image": (images[i].cpu() * std + mean),
                    "conf": conf,
                    "pred": pred_bin,
                    "true": true_bin
                }
                if pred_bin == true_bin:
                    results[bp_true]["correct"].append(entry)
                else:
                    results[bp_true]["incorrect"].append(entry)

    # Compute and normalize confusion matrix
    cm = confmat.compute().cpu().numpy()
    cm = cm / cm.sum(axis=1, keepdims=True)

    # Plot and save binary confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="GnBu")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Abnormal"])
    ax.set_yticklabels(["Normal", "Abnormal"])
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth")
    ax.set_title("Normalised confusion matrix\nfor the hierarchical two-stage classifier\n(stage one and stage two)")
    # Annotate each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout()
    fig.savefig("binary_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # # Plot top-k examples per body part
    # for bp in BODY_PARTS:
    #     # sort by confidence descending to get highest-confidence examples
    #     correct_list = sorted(
    #         results[bp]["correct"], key=lambda e: e["conf"], reverse=True
    #     )[:TOP_K]
    #     incorrect_list = sorted(
    #         results[bp]["incorrect"], key=lambda e: e["conf"], reverse=True
    #     )[:TOP_K]

    #     max_n = max(len(correct_list), len(incorrect_list))
    #     if max_n == 0:
    #         continue

    #     fig, axs = plt.subplots(2, max_n, figsize=(3 * max_n, 6))
    #     if max_n == 1:
    #         axs = axs.reshape(2, 1)

    #     # First row: correct
    #     for j in range(max_n):
    #         ax = axs[0, j]
    #         if j < len(correct_list):
    #             img = correct_list[j]["image"].permute(1, 2, 0).clamp(0, 1).numpy()
    #             ax.imshow(img)
    #             ax.set_title(f"Conf={correct_list[j]['conf']:.2f}")
    #         ax.axis("off")

    #     # Second row: incorrect
    #     for j in range(max_n):
    #         ax = axs[1, j]
    #         if j < len(incorrect_list):
    #             img = incorrect_list[j]["image"].permute(1, 2, 0).clamp(0, 1).numpy()
    #             ax.imshow(img)
    #             ax.set_title(f"Conf={incorrect_list[j]['conf']:.2f}")
    #         ax.axis("off")

    #     fig.suptitle(bp)
    #     fig.tight_layout()
    #     fig.savefig(f"{bp}_top{TOP_K}.png", dpi=300)
    #     plt.close(fig)


if __name__ == "__main__":
    main()
