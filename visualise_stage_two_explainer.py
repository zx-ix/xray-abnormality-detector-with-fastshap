import types, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

from MURA_datamodule import MURADataModule
from fastshap import ImageSurrogate, FastSHAP
from fastshap.utils import DatasetInputOnly
from unet import UNet
from stage_two_classifier import StageTwoClassifier

BODY_PART = "XR_FOREARM" # change body parts here
NUM_SAMPLES = 30
CHOSEN = [119]  # indices to plot; [] = use random sample
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize((256,256), antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
LBL_TXT = {0:"Negative",1:"Positive"}

class TupleDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        s = self.base[i]
        img = s['images']
        lbl = int(s['label'].item() if torch.is_tensor(s['label']) else s['label'])
        return img, lbl

class UNetExplainer(nn.Module):
    def __init__(self, **kw):
        super().__init__(); self.unet = UNet(n_classes=2, **kw)
    def forward(self, x):
        x = self.unet(x)
        x = F.adaptive_avg_pool2d(x, (14,14))
        return x.flatten(2).permute(0,2,1)  # (B,196,2)

class LogitLink(nn.Module):
    def forward(self, p):
        p = torch.clamp(p, 1e-6, 1-1e-6)
        return torch.log(p) - torch.log1p(-p)

dm = MURADataModule(
    dataset_location = "./MURA-v1.1",
    transforms_original = tfm,
    transforms_augment  = tfm,
    num_workers=4,
    batch_size = 1,
    body_part_filter = BODY_PART,
    test_data_split = "test",
)
dm.setup("test")
base = dm.test_dataloader().dataset
full_ds = TupleDataset(base)

rng = np.random.default_rng(SEED)
rand_idx = rng.choice(len(full_ds), size=min(NUM_SAMPLES, len(full_ds)), replace=False)
rand_idx = np.sort(rand_idx)
print("Sampled indices →", rand_idx.tolist())

sel_idx = rand_idx if not CHOSEN else np.array(CHOSEN)
print("Visualising indices →", sel_idx.tolist())


# surrogate
surr_core = torch.load(f"stage_two_surrogate_{BODY_PART}.pt", map_location='cpu').to(DEVICE).eval()
surrogate = ImageSurrogate(surr_core, width=224, height=224, superpixel_size=16)

# dummy module for pickle
if 'stage_two_explainer' not in sys.modules:
    mod = types.ModuleType('stage_two_explainer'); mod.UNetExplainer = UNetExplainer; sys.modules['stage_two_explainer'] = mod
explainer = torch.load(f"stage_two_explainer_{BODY_PART}.pt", map_location='cpu').to(DEVICE).eval()
fastshap = FastSHAP(explainer, surrogate, link = LogitLink())

# classifier
ckpt = f"./VS_code_stage_two_logs/{BODY_PART}/lightning_logs/version_0/checkpoints/{BODY_PART}_best-checkpoint.ckpt"
classifier = StageTwoClassifier.load_from_checkpoint(ckpt, map_location='cpu').to(DEVICE).eval()

images = []
shap_norm, shap_abn = [], []
true_lbls, pred_lbls, probs = [], [], []

with torch.no_grad():
    for idx in sel_idx:
        img, lbl = full_ds[int(idx)]
        x = img.unsqueeze(0).to(DEVICE)
        # shap
        shap = fastshap.shap_values(x)[0]
        if shap.ndim == 1:
            shap = shap[:, None]
        abn = shap[:, min(1, shap.shape[1]-1)].reshape(14,14)
        norm = shap[:,0].reshape(14,14) if shap.shape[1] > 1 else -abn
        shap_abn.append(abn); shap_norm.append(norm)
        # prediction
        logit = classifier(x).squeeze().item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()
        pred  = int(prob >= 0.5)
        true_lbls.append(lbl); pred_lbls.append(pred); probs.append(prob)
        # store image
        img_np = (img*STD + MEAN).clamp(0,1).permute(1,2,0).cpu().numpy()
        images.append(img_np)

if DEVICE.type == 'cuda': torch.cuda.empty_cache()

R = len(sel_idx)
fig, ax = plt.subplots(R, 3, figsize=(9, 3*R))
if R == 1: ax = np.expand_dims(ax, 0)

for i in range(R):
    # original with groundtruth and predictioon
    if LBL_TXT[true_lbls[i]] == "Positive":
        true_label = "Abnormal"
    elif LBL_TXT[true_lbls[i]] == "Negative":
        true_label = "Normal"

    if LBL_TXT[pred_lbls[i]] == "Positive":
        pred_label = "Abnormal"
    elif LBL_TXT[pred_lbls[i]] == "Negative":
        pred_label = "Normal"

    title = (
        f"Ground truth: {true_label}\n"
        f"Prediction: {pred_label}"
    )
    ax[i,0].imshow(images[i]); ax[i,0].axis('off'); ax[i,0].set_title(title, fontsize=10)
    for j, mp, t in zip([1,2], [shap_norm[i], shap_abn[i]], ["Normal", "Abnormal"]):
        print(mp)
        vmax = np.abs(mp).max(); ax[i,j].imshow(images[i]);
        ax[i,j].imshow(mp, cmap='bwr', alpha=0.5, vmin=-vmax, vmax=vmax, extent=[0,224,224,0])
        ax[i,j].axis('off'); ax[i,j].set_title(t, fontsize=9)

plt.suptitle(f"Visualising fastSHAP explanations for {BODY_PART}", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.96])
outfile = f"shap_{BODY_PART}_pred.png"
fig.savefig(outfile, dpi=300)
print("Saved →", outfile)
