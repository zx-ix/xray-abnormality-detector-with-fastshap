import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from MURA_datamodule import MURADataModule
from stage_one_classifier import StageOneClassifier
from unet import UNet
from fastshap import ImageSurrogate, FastSHAP

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
body_parts = ["XR_ELBOW","XR_FINGER","XR_FOREARM","XR_HAND", "XR_HUMERUS","XR_SHOULDER","XR_WRIST"]

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
tuple_test_dataset = TupleDataset(datamodule.test_dataloader().dataset)

surr = torch.load("stage_one_surrogate.pt", map_location=device)

surr = surr.eval()
for param in surr.parameters():
    param.requires_grad = False

surrogate = ImageSurrogate(
    surr, 
    width=224,
    height=224,
    superpixel_size=16
)

explainer = torch.load("stage_one_explainer.pt", map_location=device).eval()

fastshap  = FastSHAP(
    explainer, 
    surrogate, 
    link=nn.LogSoftmax(dim=1)
)

targets = np.array([target_int for _, target_int in tuple_test_dataset])
np.random.seed(42)

indices, labels = [], []
for body_part in range(len(body_parts)):
    index = np.where(targets == body_part)[0]
    pick = np.random.choice(index, size=3, replace=False)  # sample 3 images per body parts 
    indices.extend(pick)
    labels.extend([body_part]*3)

images, target_ints = zip(*[tuple_test_dataset[i] for i in indices])
images = torch.stack(images).to(device) # (21,3,224,224)

# compute FastSHAP values
with torch.no_grad():
    shap = fastshap.shap_values(images)

B, P, C = shap.shape
shap = shap.transpose(0,2,1).reshape(B, C, 14, 14) # (21,7,14,14)

B = images.size(0)
S_full = torch.ones(B, surrogate.num_players, device=device)
probs = surrogate(x, S_full).softmax(1).cpu().numpy()   # (21,7)

mean = torch.tensor([0.485,0.456,0.406],device=device)[:,None,None]
std  = torch.tensor([0.229,0.224,0.225],device=device)[:,None,None]
imgs = ((images*std)+mean).clamp(0,1).cpu() # (21,3,224,224)

best_model_path = "./VS_code_stage_one_logs/lightning_logs/version_0/checkpoints/best-checkpoint.ckpt"
classifier = StageOneClassifier.load_from_checkpoint(best_model_path, map_location=device).eval()

with torch.no_grad():
    logits = classifier(images) # (21,7)
probs = torch.softmax(logits, dim=1).cpu().numpy()

# plot
rows, cols = len(body_parts)*3, len(body_parts)+1        # 21*8
fig, axarr = plt.subplots(
    rows, cols,
    figsize=(cols*2.8, rows*3.0),
    gridspec_kw={"wspace":0.08, "hspace":0.3}
)

for r in range(rows):
    gt_idx    = target_ints[r]
    gt_name   = body_parts[gt_idx]
    pred_vec  = probs[r]
    pred_idx  = pred_vec.argmax()
    pred_name = body_parts[pred_idx]
    pred_prob = pred_vec[pred_idx]

    # leftmost column: original image with groundtruth and predictions
    title = (f"Ground truth: {gt_name}\n"
             f"Prediction:   {pred_name}")
    ax = axarr[r, 0]
    ax.imshow(imgs[r].permute(1,2,0))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8, loc="left")

    # remaining columns: overlay shap on top of original image
    vmax = np.abs(shap[r]).max()
    for c in range(len(body_parts)):
        sub = axarr[r, c+1]
        sub.imshow(imgs[r].permute(1,2,0))

        # overlay the shap heat-map
        sub.imshow(
            shap[r, c],
            cmap="seismic",
            vmin=-vmax, vmax=vmax,
            interpolation="nearest",
            extent=[0, 224, 224, 0],
            alpha=0.9
        )

        sub.set_axis_off()
        sub.set_xlabel(
            f"{probs[r, c]:.2f}",
            fontsize=10,
            fontweight="bold" if c == gt_idx else "normal",
            labelpad=6
        )
        if r == 0:
            sub.set_title(body_parts[c], fontsize=12, pad=6)

fig.subplots_adjust(top=0.98, bottom=0.03, left=0.03, right=0.99)
fig.savefig("stage_one_fastshap.png", dpi=300)
print("Saved: stage_one_fastshap.png")