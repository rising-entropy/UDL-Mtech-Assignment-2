"""Build the Part C figures (100-sample grids, FID/PR chart, speed chart) from results/."""
import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R = "results"; FIG = f"{R}/figures"

def grid(x, path, scale=4, gap=8):
    # x: (100, 3, 32, 32) uint8 -> 10x10 grid, nearest-neighbour upscaled, white gaps
    x = x.transpose(0, 2, 3, 1)
    s = 32 * scale; n = 10
    out = np.full((n * s + (n + 1) * gap, n * s + (n + 1) * gap, 3), 255, np.uint8)
    for i in range(100):
        r, c = divmod(i, n)
        tile = np.kron(x[i], np.ones((scale, scale, 1), np.uint8))
        out[gap + r * (s + gap): gap + r * (s + gap) + s, gap + c * (s + gap): gap + c * (s + gap) + s] = tile
    Image.fromarray(out).save(path)

def main():
    A = json.load(open(f"{R}/partA_results.json")); B = json.load(open(f"{R}/partB_results.json"))
    for name, f in [("vae_beta1", "vae_beta1_100"), ("vqvae_k256", "vqvae_k256_100"), ("wgan", "wgan_100"), ("wgan_gp", "wgan_gp_100")]:
        grid(np.load(f"{R}/samples/{f}.npy"), f"{FIG}/partC_grid_{name}.png")
    with open("data/cifar-10-batches-py/test_batch", "rb") as fh:
        t = pickle.load(fh, encoding="bytes")
    grid(t[b"data"][:100].reshape(100, 3, 32, 32), f"{FIG}/partC_grid_real.png")

    a1, a2, w, g = A["vae_beta_1"], A["vqvae_K_256"], B["wgan"], B["wgan_gp"]
    names = ["VAE\n(β=1)", "VQ-VAE\n(K=256)", "WGAN", "WGAN-GP"]
    cols = ["C0", "C2", "C1", "C3"]
    fids = [a1["FID"], a2["FID"], w["FID"], g["FID"]]; sds = [a1["FID std"], a2["FID std"], w["FID std"], g["FID std"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    b = ax[0].bar(names, fids, yerr=sds, capsize=4, color=cols)
    for r, v in zip(b, fids):
        ax[0].text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    ax[0].set_ylim(0, max(fids) * 1.12)
    ax[0].set_title("Sample quality (FID, 10k samples)"); ax[0].set_ylabel("FID (lower is better)")
    ax[0].tick_params(axis="x", labelsize=8.5)
    labels = ["VAE (β=1)", "VQ-VAE (K=256) + PixelCNN", "WGAN", "WGAN-GP"]
    for m, c, l in zip([a1, a2, w, g], cols, labels):
        ax[1].scatter(m["Recall"], m["Precision"], s=120, color=c, label=l)
    ax[1].set_xlabel("Recall (diversity)"); ax[1].set_ylabel("Precision (fidelity)"); ax[1].set_title("Precision / recall")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/partC_fid_pr.png", dpi=170); plt.close(fig)

    train_min = [a1["Train time (s)"] / 60, (a2["Train time VQ-VAE (s)"] + a2["Train time PixelCNN (s)"]) / 60,
                 w["Train time (s)"] / 60, g["Train time (s)"] / 60]
    samp_ms = [a1["Sampling (ms / 1000 img)"], a2["Generation (s / 1000 img)"] * 1000,
               w["Sampling (ms / 1000 img)"], g["Sampling (ms / 1000 img)"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(names, train_min, color=cols); ax[0].set_ylabel("minutes")
    ax[0].set_title("Training time (RTX 4090, BITS Prayogshala)")
    ax[1].bar(names, samp_ms, color=cols); ax[1].set_yscale("log"); ax[1].set_ylabel("ms per 1000 images (log)")
    ax[1].set_title("Sampling speed")
    for a in ax: a.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/partC_speed.png", dpi=170); plt.close(fig)
    print("train min", [round(x, 1) for x in train_min], "samp ms", [round(x, 1) for x in samp_ms])

if __name__ == "__main__":
    main()
