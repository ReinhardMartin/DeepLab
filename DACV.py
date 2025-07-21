import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as T
import clip
from tqdm import tqdm

# --------------------------- CONFIG ----------------------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
EPOCHS      = 10
TEMPERATURE = 0.07  # for SupCon loss and cache softmax
BETA        = 0.1   # fusion weight between CLIP and DAC logits (0‑1)

# --------------------------- DATA ------------------------------------

def get_data(data_dir="./data", transform=None):
    train = torchvision.datasets.Flowers102(root=data_dir, split="train", download=True, transform=transform)
    val   = torchvision.datasets.Flowers102(root=data_dir, split="val",   download=True, transform=transform)
    test  = torchvision.datasets.Flowers102(root=data_dir, split="test",  download=True, transform=transform)
    return train, val, test

def base_novel_categories(dataset):
    num_classes = len(set(dataset._labels))
    base = list(range(num_classes // 2))
    novel = list(range(num_classes // 2, num_classes))
    return base, novel

def split_by_classes(dataset, base_classes):
    base_idx, novel_idx = [], []
    base_set = set(base_classes)
    for i, y in enumerate(dataset._labels):
        (base_idx if y in base_set else novel_idx).append(i)
    return Subset(dataset, base_idx), Subset(dataset, novel_idx)

# --------------------------- ADAPTER ---------------------------------
class LinearAdapter(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model.eval()  # frozen
        dim = clip_model.visual.output_dim
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs)
        feats = F.normalize(self.proj(feats.float()), dim=-1)
        return feats

# --------------------------- SUPCON LOSS -----------------------------

def supcon_loss(feats, labels, temp=TEMPERATURE):
    feats = F.normalize(feats, dim=1)
    sim   = feats @ feats.t() / temp
    labels = labels.view(-1, 1)
    mask   = torch.eq(labels, labels.t()).float().to(feats.device)
    logits_mask = torch.ones_like(mask) - torch.eye(len(feats), device=feats.device)
    mask *= logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    loss = - (mask * log_prob).sum(1) / mask.sum(1).clamp_min(1e-8)
    return loss.mean()

# --------------------------- DAC CACHE -------------------------------

def build_dac_cache(model, dataset, num_classes):
    """Return W_image (N x D) and L_one_hot (N x C)."""
    keys, labels = [], []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    with torch.no_grad():
        for imgs, ys in tqdm(loader, desc="Building DAC cache"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)  # (B, D)
            keys.append(feats.cpu())
            labels.append(ys)
    W_image = torch.cat(keys)          # (N, D)
    y_all   = torch.cat(labels)        # (N,)
    L_one_hot = F.one_hot(y_all, num_classes=num_classes).float()  # (N, C)
    return W_image, L_one_hot

# --------------------------- INFERENCE -------------------------------

def dac_infer(model, W_img, L_1hot, text_feats, imgs, beta=BETA):
    """Return logits (B, C) combining DAC and CLIP text."""
    imgs = imgs.to(DEVICE)
    # CLIP text branch (frozen)
    with torch.no_grad():
        clip_img_feats = F.normalize(model.clip.encode_image(imgs), dim=-1)  # (B,D)
    clip_logits = clip_img_feats @ text_feats.T  # (B,C)

    # DAC branch
    z = model(imgs)                          # (B,D)
    w = torch.exp((W_img.to(DEVICE) @ z.T) / TEMPERATURE)  # (N,B)
    dac_logits = (L_1hot.to(DEVICE).T @ w).T   # (B,C)
    dac_logits = dac_logits / (dac_logits.sum(dim=1, keepdim=True) + 1e-8)
    # fusion
    return clip_logits + beta * dac_logits

# --------------------------- MAIN ------------------------------------
if __name__ == "__main__":
    # 1. Load CLIP
    clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)

    # 2. Data & splits
    train, val, test = get_data(transform=preprocess)
    base_cls, novel_cls = base_novel_categories(train)
    train_base, _ = split_by_classes(train, base_cls)
    val_base, _   = split_by_classes(val,   base_cls)
    test_base, test_novel = split_by_classes(test, base_cls)
    NUM_CLASSES = len(base_cls) + len(novel_cls)

    # 3. Adapter init & training
    adapter = LinearAdapter(clip_model).to(DEVICE)
    optim   = torch.optim.Adam(adapter.proj.parameters(), lr=1e-3)
    loader  = DataLoader(train_base, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        adapter.train(); total = 0
        for imgs, ys in tqdm(loader, desc=f"Epoch {epoch+1}"):
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            feats = adapter(imgs)
            loss = supcon_loss(feats, ys)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()
        print(f"Epoch {epoch+1}: SupConLoss={total/len(loader):.4f}")

    # 4. Build DAC cache using **all 10‑shot train images**
    W_img, L_1hot = build_dac_cache(adapter, train_base, NUM_CLASSES)

    # 5. Prepare CLIP text embeddings (prompt ensembling optional)
    CLASS_NAMES = train.class_to_idx.keys() if hasattr(train, "class_to_idx") else [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily",
        "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea",
        "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
        "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
        "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation",
        "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
        "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia",
        "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
        "pink-yellow dahlia?", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
        "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
        "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
        "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
        "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
        "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]

    text_in = clip.tokenize([f"a photo of the flower {c}" for c in CLASS_NAMES]).to(DEVICE)
    with torch.no_grad():
        TEXT_FEATS = F.normalize(clip_model.encode_text(text_in), dim=-1)

    # 6. Evaluation helper
    def run_split(split_loader, name="split"):
        correct, total = 0, 0
        for imgs, ys in split_loader:
            ys = ys.to(DEVICE)
            logits = dac_infer(adapter, W_img, L_1hot, TEXT_FEATS, imgs, beta=BETA)
            preds = logits.argmax(dim=1)
            correct += (preds == ys).sum().item()
            total += ys.size(0)
        print(f"{name} accuracy: {100*correct/total:.2f}%")

    print("\n--- Evaluation ---")
    run_split(DataLoader(test_base, batch_size=BATCH_SIZE),   "Test Base")
    run_split(DataLoader(test_novel, batch_size=BATCH_SIZE),  "Test Novel")
    run_split(DataLoader(test,       batch_size=BATCH_SIZE),  "Test All")