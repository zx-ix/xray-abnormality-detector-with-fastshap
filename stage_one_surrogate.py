import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from stage_one_classifier import StageOneClassifier
from MURA_datamodule import MURADataModule

from fastshap import ImageSurrogate
from fastshap.utils import MaskLayer2d, KLDivLoss, DatasetInputOnly

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
train_surr_dataset = DatasetInputOnly(tuple_train_dataset)
val_surr_dataset = DatasetInputOnly(tuple_val_dataset)

classifier = StageOneClassifier(
    download_weight=False,
    load_path="./VS_code_stage_one_logs/lightning_logs/version_0/checkpoints/best-checkpoint.ckpt",
    target_type="multiclass",
    output_dim=7,
    checkpoint_metric="CohenKappa",
    optim_type="AdamW",
    learning_rate=1e-4,
    weight_decay=1e-5,
    decay_power="cosine",
    warmup_steps=1000,
    batch_size=batch_size,
    distillation=False,
    loss_weight=None,
).to(device)

classifier.eval()
for parameter in classifier.parameters():
    parameter.requires_grad = False

original_model = nn.Sequential(
    classifier,
    nn.Softmax(dim=1)
)

classifier_surr = StageOneClassifier(
    download_weight=False,
    load_path="./VS_code_stage_one_logs/lightning_logs/version_0/checkpoints/best-checkpoint.ckpt",
    target_type="multiclass",
    output_dim=7,
    checkpoint_metric="CohenKappa",
    optim_type="AdamW",
    learning_rate=1e-4,
    weight_decay=1e-5,
    decay_power="cosine",
    warmup_steps=1000,
    batch_size=batch_size,
    distillation=False,
    loss_weight=None,
).to(device)

surr = nn.Sequential(
    MaskLayer2d(value=0, append=False),
    classifier_surr
).to(device)

surrogate = ImageSurrogate(
    surr,
    width=224,
    height=224,
    superpixel_size=16
)

# train the surrogate
surrogate.train_original_model(
    train_surr_dataset,
    val_surr_dataset,
    original_model,
    batch_size=batch_size,
    max_epochs=50,
    loss_fn=KLDivLoss(),
    lookback=10,
    bar=True,
    verbose=True
)

# save surrogate
surr.cpu()
torch.save(surr, 'stage_one_surrogate.pt')
surr.to(device)