import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path(__file__).parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path(__file__).parent / "output"
GENERATED_DIR = OUTPUT_DIR / "generated"
MATCH_RESULT_PATH = OUTPUT_DIR / "match_results.csv"
MATCH_RESULT_SSIM_PATH = OUTPUT_DIR / "match_results_ssim.csv"
MATCH_TRAIN_RESULT_PATH = OUTPUT_DIR / "match_train_results.csv"
MATCH_TRAIN_RESULT_SSIM_PATH = OUTPUT_DIR / "match_train_results_ssim.csv"


def get_first_image_size(root_dir: Path):
    # 取一张样本图片，获取原始宽高
    files = sorted(
        [p for p in root_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    if not files:
        raise RuntimeError(f"No image files found in {root_dir}")
    img = Image.open(files[0]).convert("RGB")
    return img.size  # (width, height)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        # 只保留常见图片格式
        self.files = sorted(
            [p for p in self.root_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        # 训练时统一缩放到方形尺寸
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), str(img_path.name)


class ConvAutoencoder(nn.Module):
    def __init__(self, image_size=64, latent_dim=64):
        super().__init__()
        assert image_size % 8 == 0, "image_size 必须能被 8 整除，以匹配三次下采样/上采样结构"
        self.image_size = image_size
        # 三次 stride=2 卷积后的空间尺寸
        self.spatial = image_size // 8

        # 编码部分：三次下采样，通道数逐步增加
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * self.spatial * self.spatial, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * self.spatial * self.spatial)
        # 解码部分：三次上采样，恢复到原通道数
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, self.spatial, self.spatial)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


def train_model(
    epochs=200,
    batch_size=8,
    lr=1e-3,
    latent_dim=256,
    image_size=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 如果未指定 image_size，则根据原始图片尺寸自动设置
    orig_w, orig_h = get_first_image_size(TRAIN_DIR)
    print(f"Original image size in train: {orig_w}x{orig_h}")
    if image_size is None:
        # 取较小边，并向下调整到能被 8 整除
        base = min(orig_w, orig_h)
        image_size = max(64, base - (base % 8))
    print(f"Training image_size (square, resized): {image_size}x{image_size}")

    train_set = ImageFolderDataset(TRAIN_DIR, image_size=image_size)
    test_set = ImageFolderDataset(TEST_DIR, image_size=image_size)
    n_train = len(train_set)
    n_test = len(test_set)
    print(f"Train images: {n_train} (data/train), Test images: {n_test} (data/test)")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ConvAutoencoder(image_size=image_size, latent_dim=latent_dim).to(device)
    # 重建损失：MSE + L1
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        # 线性下降学习率：从 lr 逐步降到 1e-5
        end_lr = 1e-5
        decay_ratio = (epoch - 1) / max(1, epochs - 1)
        current_lr = lr + (end_lr - lr) * decay_ratio
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = 0.5 * mse_loss(recon, x) + 0.5 * l1_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / n_train
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.6f} - LR: {current_lr:.6e}")

    # 保存模型和与尺寸相关的信息
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "train_image_size": image_size,
            "orig_width": orig_w,
            "orig_height": orig_h,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # 保存测试集文件名列表
    test_names = test_set.files
    split_path = OUTPUT_DIR / "split_info.txt"
    with open(split_path, "w", encoding="utf-8") as f:
        f.write("Test files:\n")
        for p in test_names:
            f.write(f"{p.name}\n")
    print(f"Test split info saved to {split_path}")

    # 保存测试集图像张量和文件名，用于后续匹配
    test_images = []
    test_filenames = []
    for x, name in test_loader:
        test_images.append(x)
        test_filenames.extend(name)
    test_images = torch.cat(test_images, dim=0)
    torch.save(
        {"images": test_images, "filenames": test_filenames},
        OUTPUT_DIR / "test_tensors.pt",
    )
    print("Test tensors saved.")

    # 保存一批训练集的重建结果，方便查看训练效果
    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(train_loader))
        x_batch = x_batch.to(device)
        recon_batch = model(x_batch)

    recon_dir = OUTPUT_DIR / "recon_samples"
    recon_dir.mkdir(exist_ok=True)
    num_samples = min(8, x_batch.size(0))
    for i in range(num_samples):
        # 原图
        orig = x_batch[i].cpu().numpy()
        orig = np.transpose(orig, (1, 2, 0))
        orig = (orig * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(orig).save(recon_dir / f"train_{i}_orig.png")

        # 重建图
        rec = recon_batch[i].cpu().numpy()
        rec = np.transpose(rec, (1, 2, 0))
        rec = (rec * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(rec).save(recon_dir / f"train_{i}_recon.png")

    print(f"Saved {num_samples} reconstruction samples to {recon_dir}")


def generate_images(num_images=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(OUTPUT_DIR / "autoencoder.pt", map_location=device)
    latent_dim = checkpoint["latent_dim"]
    train_image_size = checkpoint.get("train_image_size", 64)
    orig_w = checkpoint.get("orig_width", train_image_size)
    orig_h = checkpoint.get("orig_height", train_image_size)

    model = ConvAutoencoder(image_size=train_image_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 基于训练集得到一批潜在向量 z
    base_dataset = ImageFolderDataset(TRAIN_DIR, image_size=train_image_size)
    base_loader = DataLoader(base_dataset, batch_size=16, shuffle=False)

    all_z = []
    with torch.no_grad():
        for x, _ in base_loader:
            x = x.to(device)
            z = model.encode(x)
            all_z.append(z.cpu())
    if not all_z:
        print("No images found for latent sampling.")
        return
    all_z = torch.cat(all_z, dim=0)  # [N, latent_dim]

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # 在训练集的 z 附近添加扰动采样
    noise_scale = 0.1
    with torch.no_grad():
        n_base = all_z.size(0)
        for i in range(num_images):
            idx = np.random.randint(0, n_base)
            z_base = all_z[idx : idx + 1].to(device)
            noise = torch.randn_like(z_base) * noise_scale
            z = z_base + noise

            img = model.decode(z).cpu().numpy()[0]
            # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(img)
            # 缩放回原始尺寸
            im = im.resize((orig_w, orig_h), Image.BILINEAR)
            im.save(GENERATED_DIR / f"gen_{i:04d}.png")
    print(f"Generated {num_images} images in {GENERATED_DIR}")


def compare_generated_with_test(threshold=1e-2):
    # 每个生成样本对每张测试集图像计算 MSE，保存为 CSV
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入测试集数据
    test_data = torch.load(OUTPUT_DIR / "test_tensors.pt", map_location=device)
    test_images = test_data["images"].to(device)  # [N, C, H, W]
    test_filenames = test_data["filenames"]
    n_test = test_images.size(0)

    gen_files = sorted([p for p in GENERATED_DIR.iterdir() if p.suffix.lower() == ".png"])
    if not gen_files:
        print("No generated images found. Please run generation first.")
        return

    # 每行：生成样本名 + 对每张测试图的 MSE
    rows_mse = []

    # SSIM 矩阵：每个生成样本对每一张测试样本都计算一次 SSIM
    rows_ssim = []
    with torch.no_grad():
        for gen_path in gen_files:
            img = Image.open(gen_path).convert("RGB")
            img = img.resize((test_images.size(3), test_images.size(2)))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            gen_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1,C,H,W]

            diff = (test_images - gen_tensor) ** 2
            mse = diff.view(n_test, -1).mean(dim=1)  # [N]
            rows_mse.append([gen_path.name] + [f"{mse[i].item():.8f}" for i in range(n_test)])

            g_np = gen_tensor[0].cpu().numpy().transpose(1, 2, 0)
            ssim_row = []
            for i in range(n_test):
                t_np = test_images[i].cpu().numpy().transpose(1, 2, 0)
                s = structural_similarity(g_np, t_np, channel_axis=2, data_range=1.0)
                ssim_row.append(f"{s:.6f}")
            rows_ssim.append([gen_path.name] + ssim_row)

    MATCH_RESULT_PATH.parent.mkdir(exist_ok=True)
    with open(MATCH_RESULT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generated"] + test_filenames)
        writer.writerows(rows_mse)

    with open(MATCH_RESULT_SSIM_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generated"] + test_filenames)
        writer.writerows(rows_ssim)

    print(f"Matching results (MSE) saved to {MATCH_RESULT_PATH}")
    print(f"Matching results (SSIM matrix) saved to {MATCH_RESULT_SSIM_PATH}")


def compare_generated_with_train(threshold=1e-3):
    # 每个生成样本对每张训练集图像计算 MSE，保存为 CSV
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_w, orig_h = get_first_image_size(TRAIN_DIR)
    base = min(orig_w, orig_h)
    image_size = max(64, base - (base % 8))

    train_set = ImageFolderDataset(TRAIN_DIR, image_size=image_size)
    if len(train_set) == 0:
        print("No images found in data/train.")
        return
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)

    train_images = []
    train_filenames = []
    for x, name in train_loader:
        train_images.append(x)
        train_filenames.extend(name)
    if not train_images:
        print("No train images after split.")
        return
    train_images = torch.cat(train_images, dim=0).to(device)
    n_train = train_images.size(0)

    gen_files = sorted([p for p in GENERATED_DIR.iterdir() if p.suffix.lower() == ".png"])
    if not gen_files:
        print("No generated images found. Please run generation first.")
        return

    # 每行：生成样本名 + 对每张训练图的 MSE
    rows_mse = []

    # SSIM 矩阵：每个生成样本对每一张训练样本都计算一次 SSIM
    rows_ssim = []
    with torch.no_grad():
        for gen_path in gen_files:
            img = Image.open(gen_path).convert("RGB")
            img = img.resize((train_images.size(3), train_images.size(2)))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            gen_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1,C,H,W]

            diff = (train_images - gen_tensor) ** 2
            mse = diff.view(n_train, -1).mean(dim=1)
            rows_mse.append([gen_path.name] + [f"{mse[i].item():.8f}" for i in range(n_train)])

            g_np = gen_tensor[0].cpu().numpy().transpose(1, 2, 0)
            ssim_row = []
            for i in range(n_train):
                t_np = train_images[i].cpu().numpy().transpose(1, 2, 0)
                s = structural_similarity(g_np, t_np, channel_axis=2, data_range=1.0)
                ssim_row.append(f"{s:.6f}")
            rows_ssim.append([gen_path.name] + ssim_row)

    MATCH_TRAIN_RESULT_PATH.parent.mkdir(exist_ok=True)
    with open(MATCH_TRAIN_RESULT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generated"] + [Path(n).name for n in train_filenames])
        writer.writerows(rows_mse)

    with open(MATCH_TRAIN_RESULT_SSIM_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generated"] + [Path(n).name for n in train_filenames])
        writer.writerows(rows_ssim)

    print(f"Train matching results (MSE) saved to {MATCH_TRAIN_RESULT_PATH}")
    print(f"Train matching results (SSIM matrix) saved to {MATCH_TRAIN_RESULT_SSIM_PATH}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train AE, generate images and compare.")
    parser.add_argument("--mode", type=str, default="all", choices=["train", "gen", "match", "all"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_gen", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        train_model(epochs=args.epochs)
    if args.mode in ["gen", "all"]:
        generate_images(num_images=args.num_gen)
    if args.mode in ["match", "all"]:
        compare_generated_with_test(threshold=args.threshold)
        compare_generated_with_train(threshold=args.threshold)


if __name__ == "__main__":
    main()


