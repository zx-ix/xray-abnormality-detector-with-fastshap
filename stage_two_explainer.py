import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from MURA_datamodule import MURADataModule
from unet import UNet
from fastshap import ImageSurrogate, FastSHAP
from fastshap.utils import DatasetInputOnly

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

class UNetExplainer(nn.Module):
    def __init__(self, *, n_players: int = 196, n_classes: int = 1, **unet_kwargs):
        super().__init__()
        self.unet = UNet(n_classes=n_classes, **unet_kwargs)
        self.n_players = n_players
        self.n_classes = n_classes

    def forward(self, x):
        out = self.unet(x)
        out = F.adaptive_avg_pool2d(out, (14, 14))
        out = out.flatten(2).permute(0, 2, 1)
        return out

# link function: logit for Bernoulli distribution
class LogitLink(nn.Module):
    def forward(self, x: torch.Tensor):
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return torch.log(x) - torch.log1p(-x)  # log(p/(1-p))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

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

surrogate_pattern = "stage_two_surrogate_{bp}.pt"

for bp in body_parts:
    print(f"\nTraining explainer for {bp}:")

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

    train_exp_ds = DatasetInputOnly(TupleDataset(dm.train_dataloader().dataset))
    val_exp_ds = DatasetInputOnly(TupleDataset(dm.val_dataloader().dataset))

    surr_net = torch.load(surrogate_pattern.format(bp=bp), map_location=device)
    surr_net.eval()
    for p in surr_net.parameters():
        p.requires_grad = False

    surrogate = ImageSurrogate(
        surr_net,
        width224,
        height=224,
        superpixel_size=16
    )

    explainer = UNetExplainer(
        n_players=196,
        n_classes=1,
        num_down=4,
        num_up=3,
        num_convs=2
    ).to(device)

    fastshap = FastSHAP(
        explainer,
        surrogate,
        link=LogitLink()
    )

    fastshap.train(
        train_exp_ds,
        val_exp_ds,
        batch_size=batch_size,
        num_samples=2,
        max_epochs=50,
        eff_lambda=1e-2,
        validation_samples=1,
        lookback=10,
        bar=True,
        verbose=True
    )

    explainer.cpu()
    save_name = f"stage_two_explainer_{bp}.pt"
    torch.save(explainer, save_name)
    print(f"Saved explainer for {bp}: {save_name}")
    explainer.to(device)