import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from stage_two_classifier import StageTwoClassifier
from MURA_datamodule import MURADataModule

from fastshap import ImageSurrogate
from fastshap.utils import MaskLayer2d, DatasetInputOnly

class TupleDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample["images"]
        target = sample["label"]
        return image, torch.tensor(target, dtype=torch.long)

def bernoulli_kl_div(original_logits, surr_logits):
    p = torch.sigmoid(original_logits).clamp(min=1e-6, max=1-1e-6)
    q = torch.sigmoid(surr_logits).clamp(min=1e-6, max=1-1e-6)
    kl = p*torch.log(p/q) + (1-p)*torch.log((1-p)/(1-q))
    return kl.mean()

os.environ["QT_QPA_PLATFORM"] = "offscreen"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

augmenter = transforms.RandAugment(
    num_ops=2,
    magnitude=9,
    num_magnitude_bins=31,
    interpolation=transforms.InterpolationMode.NEAREST,
    fill=None
)

transform_original = transforms.Compose([
    transforms.Resize((256,256), antialias=True),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

transform_augment = transforms.Compose([
    transforms.Resize((256,256), antialias=True),
    augmenter,
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    transforms.RandomErasing()
])

body_parts = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND',  'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

ckpt_pattern = (
    "./VS_code_stage_two_logs/{bp}/"
    "lightning_logs/version_0/checkpoints/{bp}_best-checkpoint.ckpt"
)

for bp in body_parts:
    print(f"\nSurrogate for {bp}:")

    dm = MURADataModule(
        dataset_location="./MURA-v1.1",
        transforms_original=transform_original,
        transforms_augment=transform_augment,
        num_workers=16,
        batch_size=batch_size,
        body_part_filter=bp,
        test_data_split="test"
    )
    dm.setup("fit")

    train_ds = DatasetInputOnly(TupleDataset(dm.train_dataloader().dataset))
    val_ds = DatasetInputOnly(TupleDataset(dm.val_dataloader().dataset))

    ckpt_path = ckpt_pattern.format(bp=bp)
    clf = StageTwoClassifier(
        download_weight=False,
        load_path=ckpt_path,
        target_type="binary",
        output_dim=1,
        checkpoint_metric="CohenKappa",
        optim_type="AdamW",
        learning_rate=1e-4,
        weight_decay=1e-5,
        decay_power="cosine",
        warmup_steps=1000,
        batch_size=batch_size,
        distillation=False,
        body_part_filter=bp,
        loss_weight=None
    ).to(device)

    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    original_model = nn.Sequential(clf, nn.Sigmoid())

    clf_surr = StageTwoClassifier(
        download_weight=False,
        load_path=ckpt_path,
        target_type="binary",
        output_dim=1,
        checkpoint_metric="CohenKappa",
        optim_type="AdamW",
        learning_rate=1e-4,
        weight_decay=1e-5,
        decay_power="cosine",
        warmup_steps=1000,
        batch_size=batch_size,
        distillation=False,
        body_part_filter=bp,
        loss_weight=None
    ).to(device)

    surr_net = nn.Sequential(
        MaskLayer2d(value=0, append=False),
        clf_surr,
        nn.Sigmoid()
    ).to(device)

    surrogate = ImageSurrogate(
        surr_net,
        width=224,
        height=224,
        superpixel_size = 16
    )

    surrogate.train_original_model(
        train_ds,
        val_ds,
        original_model,
        batch_size=batch_size,
        max_epochs=50,
        loss_fn=bernoulli_kl_div,
        lookback=10,
        bar=True,
        verbose=True
    )

    surr_net.cpu()
    save_path = f"stage_two_surrogate_{bp}.pt"
    torch.save(surr_net, save_path)
    print(f"Saved surrogate for {bp}: {save_path}")
    surr_net.to(device)