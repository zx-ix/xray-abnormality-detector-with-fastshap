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

class UNetExplainer(nn.Module):
    def __init__(self, *, n_players, n_classes, **unet_kwargs):
        super().__init__()
        self.unet = UNet(n_classes=n_classes, **unet_kwargs)
        self.n_players = n_players # 96
        self.n_classes = n_classes # 7

    def forward(self, x):
        out = self.unet(x) # (B,7,112,112)
        out = F.adaptive_avg_pool2d(out, (14, 14)) # match superpixel (B,7,14,14)
        out = out.flatten(2).permute(0, 2, 1) # flatten to (B,196,7)

        return out

class TupleDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample["images"] # (3,224,2240 tensor
        target = sample["body_part_idx"] # (7) one-hot tensor
        target_int  = torch.argmax(target).long() # scalar index
        return image, target_int

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
])

datamodule = MURADataModule(
    dataset_location="./MURA-v1.1",
    transforms_original=transform_original,
    transforms_augment=transform_augment,
    num_workers=16,
    batch_size=batch_size,
    body_part_filter=None,
    test_data_split="test",
)
datamodule.setup("fit")

# wrap for tuple (x,y)
tuple_train_dataset = TupleDataset(datamodule.train_dataloader().dataset)
tuple_val_dataset = TupleDataset(datamodule.val_dataloader().dataset)

# set up datasets
train_exp_dataset = DatasetInputOnly(tuple_train_dataset)
val_exp_dataset = DatasetInputOnly(tuple_val_dataset)

surr = torch.load('stage_one_surrogate.pt').to(device)

surr = surr.eval()
for param in surr.parameters():
    param.requires_grad = False

surrogate = ImageSurrogate(
    surr,
    width=224,
    height=224,
    superpixel_size=16
)

explainer = UNetExplainer(
    n_players=196,
    n_classes=7,
    num_down=4,
    num_up=3,
    num_convs=2
).to(device)

fastshap = FastSHAP(
    explainer, 
    surrogate, 
    link=nn.LogSoftmax(dim=1)
)

fastshap.train(
    train_exp_dataset,
    val_exp_dataset,
    batch_size=16,
    num_samples=2,
    max_epochs=100,
    eff_lambda=1e-2,
    validation_samples=1,
    lookback=10,
    bar=True,
    verbose=True)

explainer.cpu()
torch.save(explainer, 'stage_one_explainer.pt')
explainer.to(device)