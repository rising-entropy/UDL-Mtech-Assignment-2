"""Build udl_assignment2_partC_group6.pdf from results/ (run tools/partC_figures.py first).

Usage (from the repo root):
    python tools/partC_figures.py
    python tools/build_partC.py
The scores table is read from results/partA_results.json and results/partB_results.json;
the prose below quotes the same numbers and must be revised by hand after a re-run.
"""
import json, os
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

OUT = "udl_assignment2_partC_group6.pdf"
FIG = "results/figures"
A = json.load(open("results/partA_results.json"))
B = json.load(open("results/partB_results.json"))

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.5, leading=13.2, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=15, leading=19, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, fontSize=10.5, leading=14, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
cap = ParagraphStyle("cap", parent=body, fontName="DV-B", fontSize=10, leading=13, spaceBefore=4, spaceAfter=3)

W = A4[0] - 3.6 * cm


def P(t, s=body):
    return Paragraph(t, s)


def img(path, width):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(path, width=width, height=width * h / w)


def tbl(rows, widths, head_bg="#2f4f6f", font=8.6):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "DV", font),
        ("FONT", (0, 0), (-1, 0), "DV-B", font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(head_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f3f6")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def scores_rows():
    rows = [["Model", "PSNR (dB)", "FID ↓", "Prec. ↑", "Recall ↑", "Train (min)", "ms / 1k img"]]
    for b in (1, 2, 4, 10):
        m = A[f"vae_beta_{b}"]
        rows.append([f"β-VAE (β={b})", f"{m['Test PSNR (dB)']:.2f}", f"{m['FID']:.1f} ± {m['FID std']:.1f}",
                     f"{m['Precision']:.2f}", f"{m['Recall']:.3f}", f"{m['Train time (s)'] / 60:.1f}",
                     f"{m['Sampling (ms / 1000 img)']:.1f}"])
    for k in (512, 256, 128):
        m = A[f"vqvae_K_{k}"]
        rows.append([f"VQ-VAE + PixelCNN (K={k})", f"{m['Test PSNR (dB)']:.2f}", f"{m['FID']:.1f} ± {m['FID std']:.1f}",
                     f"{m['Precision']:.2f}", f"{m['Recall']:.3f}",
                     f"{(m['Train time VQ-VAE (s)'] + m['Train time PixelCNN (s)']) / 60:.1f}",
                     f"{m['Generation (s / 1000 img)'] * 1000:.0f}"])
    for key, name in (("wgan", "WGAN"), ("wgan_gp", "WGAN-GP")):
        m = B[key]
        rows.append([name, "n/a", f"{m['FID']:.1f} ± {m['FID std']:.1f}", f"{m['Precision']:.2f}",
                     f"{m['Recall']:.3f}", f"{m['Train time (s)'] / 60:.1f}", f"{m['Sampling (ms / 1000 img)']:.1f}"])
    return rows


def build():
    s = []
    s.append(P("Unsupervised Deep Learning — Assignment 2", h1))
    s.append(P("Part C: Comparative analysis of (β-)VAE, VQ-VAE + PixelCNN, WGAN and WGAN-GP on CIFAR-10", sub))
    s.append(Spacer(1, 6))
    group = [["Name", "BITS ID", "Contribution %"],
             ["Brincita Prima Cutinha", "2025AG05256", "100"],
             ["Devang Ajay Kamble", "2025AE05173", "100"],
             ["Saksham Jindal", "2025AE05983", "100"],
             ["Shantanu Biswas", "2025AE05079", "100"],
             ["Ashish Chauhan", "2025AG05928", "100"]]
    s.append(tbl(group, [6.5 * cm, 4 * cm, 3.5 * cm]))
    s.append(P("Group 6", small))

    # 1. Setup
    s.append(P("1. Setup and evaluation protocol", h2))
    s.append(P(
        "All four model families were trained on CIFAR-10 (50,000 training and 10,000 test images, 32x32x3) with pixel "
        "values normalised to [0, 1]. Part A (the (β-)VAE and the VQ-VAE with a PixelCNN prior) and Part B (WGAN and "
        "WGAN-GP) are in the two submitted notebooks; this report only compares their results. "
        "<b>Metrics.</b> Reconstruction is measured as the mean per-image PSNR on the 10,000 test images (only the "
        "autoencoders have an encoder, so PSNR is not defined for the GANs). Sample quality is measured with the Frechet "
        "Inception Distance (FID, Inception-v3 pool3 features with the standard TF-FID weights, reference = the CIFAR-10 "
        "training set), computed on 10,000 generated images and reported as mean ± std over 3 independent sampling "
        "rounds. As an extra we report k-NN precision (fidelity) and recall (diversity) on the same features. Speed is "
        "wall-clock training time and the time needed to generate 1,000 images. "
        "<b>Budgets.</b> VAE: 40 epochs per β; VQ-VAE: 40 epochs per K plus 40 epochs for each PixelCNN prior; WGAN and "
        "WGAN-GP: 25,000 generator iterations with 5 critic steps each (about 160 critic epochs). All numbers come from "
        "a single run per model (seed 42)."))
    s.append(P("<b>Compute infrastructure (BITS Prayogshala).</b> All training and evaluation runs reported here were "
               "executed on the BITS Prayogshala GPU infrastructure with the following configuration; verification "
               "screenshots are submitted separately.", body))
    infra = [["Component", "Configuration"],
             ["GPU", "1x NVIDIA GeForce RTX 4090, 24 GB GDDR6X (driver 595.91.07)"],
             ["CPU / memory", "AMD EPYC 7K62 48-core host; 12 vCPUs and 31 GB RAM allocated to the job"],
             ["OS / container", "Ubuntu 24.04.3 LTS (PyTorch CUDA 12.8 container image)"],
             ["CUDA / cuDNN", "CUDA 12.8, cuDNN 9.10.2"],
             ["Python stack", "Python 3.12.3, PyTorch 2.8.0+cu128, torchvision 0.23.0, NumPy 2.1.2, SciPy 1.15.3"],
             ["FID implementation", "pytorch-fid 0.3.0 (Inception-v3 pool3, TF-FID weights)"],
             ["Execution", "Jupyter notebooks run top-to-bottom; outputs embedded in the submitted .ipynb / .html"]]
    t = tbl(infra, [3.6 * cm, W - 3.6 * cm], font=8.3)
    t.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "LEFT")]))
    s.append(t)

    # 2. Scores
    s.append(PageBreak())
    s.append(KeepTogether([P("2. Evaluation scores", h2), P("The table lists every trained model; the four models that are compared in the rest of this report are "
               "the β = 1 VAE, the K = 256 VQ-VAE with PixelCNN prior, WGAN and WGAN-GP."),
    tbl(scores_rows(), [4.5 * cm, 2.0 * cm, 2.5 * cm, 1.6 * cm, 1.8 * cm, 2.2 * cm, 2.4 * cm]),
    P("FID: mean ± std over 3 independent rounds of 10,000 samples, Inception-v3 pool3 features, reference = "
               "CIFAR-10 training set. PSNR is not defined for the GANs (no encoder). VQ-VAE training time is the sum of "
               "the autoencoder and the PixelCNN prior. Sampling time is for 1,000 images on the GPU (VQ-VAE includes "
               "the 64 sequential PixelCNN steps). All times measured on the RTX 4090 described above.", small)]))
    s.append(img(f"{FIG}/partC_fid_pr.png", W))
    s.append(P(
        "<b>Ranking by FID</b> (lower is better): <b>WGAN-GP (63.2) &lt; VQ-VAE + PixelCNN (K = 256: 83.1; 80-83 over "
        "all K) &lt; WGAN (116.2) &lt; β-VAE (β = 1: 141.8)</b>. The ordering is the one reported in the literature for "
        "these model families: pixel-wise VAEs are the blurriest, a discrete latent model with an autoregressive prior "
        "is much sharper, and an adversarially trained generator with a well-conditioned critic is the best of the "
        "four, whereas weight clipping gives up much of the GAN advantage. The differences between the families (about "
        "17 to 79 FID points) are far larger than the evaluation noise (± 0.2-0.4). Precision and recall show a "
        "consistent picture for the three best models (WGAN-GP: 0.57 / 0.15, VQ-VAE: 0.57 / 0.11, VAE: 0.26 / 0.006), "
        "while the precision of WGAN (0.63) is high although its recall is nearly zero (0.007); this is a typical "
        "artefact of the k-NN estimate when the generated set is concentrated in a dense region, so for WGAN we rely on "
        "FID and the pictures rather than on precision."))

    # 3. Samples
    s.append(PageBreak())
    s.append(P("3. 100 visual examples per model and perceptual quality", h2))
    s.append(P("Each grid below shows the first 100 samples of a fixed random draw (no cherry-picking) of the "
               "corresponding model, enlarged. The real CIFAR-10 test images are shown at the end for reference. "
               "Perceptual comments are given under each grid."))
    gw = 8.6 * cm
    grids = [
        ("vae_beta1", "VAE (β=1) — 100 random samples",
         "<b>Perceptual quality: low.</b> Every sample is a smooth, low-frequency blob: the global colour layout and a "
         "rough 'object in front of a background' composition are right (a few ships on blue water and car-like shapes "
         "are guessable), but there are no edges, no texture and no recognisable object. The colours are slightly "
         "desaturated and the samples are less diverse than the real images. This is the typical VAE result of a "
         "pixel-wise likelihood that predicts the average of all images compatible with a latent code."),
        ("vqvae_k256", "VQ-VAE (K=256) + PixelCNN — 100 random samples",
         "<b>Perceptual quality: medium to good.</b> Crisp edges, high-contrast textures and a wide range of colours; "
         "scenes with sky, ground, vehicles and animals are often recognisable and the shading is locally consistent. "
         "The weaknesses are global: object shapes frequently do not close into a clean object and some textures look "
         "patchy. Clearly sharper and more diverse than the VAE."),
        ("wgan", "WGAN — 100 random samples",
         "<b>Perceptual quality: low to medium.</b> Smeared, washed-out colour patches with noisy local texture and a "
         "plausible sky-above-ground layout; almost no object structure, and many samples share the same composition "
         "(light upper part, dark blob in the middle), which is consistent with its near-zero recall. Sharper than the "
         "VAE, but less coherent than the VQ-VAE."),
        ("wgan_gp", "WGAN-GP — 100 random samples",
         "<b>Perceptual quality: best of the four.</b> The most varied set, with sharp edges, consistent lighting and "
         "many recognisable vehicles, animals and ships on natural backgrounds. Objects are still distorted (typical for "
         "a small DCGAN trained for 25k iterations) and some samples contain local artefacts, but its object shapes look "
         "at least as coherent as those of the VQ-VAE samples, with higher variety."),
    ]
    for key, title, text in grids:
        s.append(KeepTogether([P(title, cap), img(f"{FIG}/partC_grid_{key}.png", gw), Spacer(1, 3), P(text)]))
    s.append(KeepTogether([P("Reference: 100 real CIFAR-10 test images", cap), img(f"{FIG}/partC_grid_real.png", gw)]))

    # 4. Discussion
    s.append(PageBreak())
    s.append(P("4. Comparative discussion", h2))
    s.append(P(
        "<b>Reconstruction error.</b> Only the two autoencoder families can reconstruct. The VQ-VAE reconstructs much "
        "better than the VAE: 24.86 dB (K = 256; 24.3-25.4 dB for K = 128-512) against 20.69 dB (β = 1), which is a "
        "2.6x lower mean squared error, even though its code (64 discrete symbols, 512 bits for K = 256) is far smaller "
        "than the VAE's 128-dimensional real vector. A spatial grid of discrete codes preserves local structure that a "
        "single global Gaussian code cannot. Within each family reconstruction follows the size of the bottleneck: PSNR "
        "falls monotonically as β grows (20.69, 19.39, 18.23, 16.81 dB for β = 1, 2, 4, 10, because the KL term forces "
        "the code to carry less information) and rises with the codebook size (24.28, 24.86, 25.43 dB for K = 128, "
        "256, 512; about 0.6 dB per doubling). The GANs have no encoder and therefore no reconstruction error; their "
        "latent codes can only be used for generation."))
    s.append(P(
        "<b>Sampling quality.</b> For unconditional sampling the order is the FID order above. Decoder-only sampling "
        "from the VAE prior is the weakest because the aggregate posterior does not match N(0, I) and the decoder "
        "averages: raising β makes the latent space more regular but the samples worse (FID 141.8, 172.9, 217.3, "
        "277.6), as the model switches off latent dimensions (123, 81, 44 and 16 of 128 remain active). The VQ-VAE "
        "solves the prior problem by learning it: sampling uniformly random codes gives texture noise, whereas "
        "PixelCNN-sampled codes give coherent scenes; its FID is nearly flat over K (80.3, 83.1, 80.6 for K = 512, 256, "
        "128) because it is limited by the prior, not by the codec. The adversarial models optimise sample realism "
        "directly: WGAN-GP reaches 63.2 and its recall of 0.15 is the highest of all models, while weight clipping "
        "(116.2) limits the critic to a few effective parameters (45 % of its weights sit on the clipping boundary at "
        "the end of training) and its samples are much less diverse."))
    s.append(img(f"{FIG}/partC_speed.png", W * 0.8))
    s.append(P(
        "<b>Speed.</b> Training: the VAE is the cheapest (about 1.9 min); the VQ-VAE pipeline is two-stage and takes "
        "about 6.2 min for K = 256 (2.3 min for the autoencoder plus 3.8 min for the PixelCNN); WGAN takes 12.0 min and "
        "WGAN-GP 21.1 min (the gradient penalty needs a double back-propagation, 1.75x the time per iteration). "
        "Sampling: the VAE (about 2 ms per 1,000 images) and the GANs (about 12 ms; the two generators are identical, "
        "so their times agree) generate in a single forward pass, whereas the PixelCNN has to sample the 64 latent "
        "positions one after the other, which takes about 237 ms per 1,000 images, i.e. roughly 20x slower than the "
        "GANs and 125x slower than the VAE. In short: the VAE is the fastest model but the worst generator; WGAN-GP "
        "gives the best quality at the highest training cost but with fast sampling; VQ-VAE + PixelCNN is a good second "
        "in quality, cheap to train, but the slowest sampler."))
    s.append(P(
        "<b>Training stability and practical behaviour.</b> The VAE and VQ-VAE are trained with a single, stable loss "
        "and their training curves are smooth and monotone. The VQ-VAE needs one extra ingredient: without a remedy "
        "randomly initialised codes are easily left unused (codebook collapse), so we use EMA codebook updates and "
        "dead-code restarts, after which all codebooks reach 100 % usage with a perplexity of 80-88 % of K. The "
        "PixelCNN priors overfit mildly (test NLL about 0.3 nats per code above training NLL). The GANs are more "
        "delicate: the WGAN-GP critic gradient norm fell from 1.46 to 1.09 (target 1) and its FID fell smoothly (and was "
        "still falling after 25,000 iterations, so more training would help), whereas the clipped WGAN has a noisy FID "
        "curve (it rises at iterations 17,500 and 22,500) and a critic whose weights collapse onto the two boundary "
        "values. The Wasserstein estimate fell steadily for both and tracked sample quality, which makes the critic loss "
        "a useful training monitor."))

    # 5. Ablations
    s.append(P("5. Ablation highlights from Parts A and B", h2))
    for f, c in (("vae_psnr_vs_epoch", "β-VAE: test PSNR, KL and reconstruction loss versus epoch."),
                 ("vqvae_k_comparison", "VQ-VAE + PixelCNN: effect of the codebook size K."),
                 ("gan_training_dynamics", "WGAN vs WGAN-GP: Wasserstein estimate, FID during training, generator loss.")):
        s.append(KeepTogether([img(f"{FIG}/{f}.png", W * 0.92), P(c, small)]))
    s.append(P(
        "<b>β-VAE (Part A-1):</b> a larger β gives a more compact, more regular latent space but loses reconstruction "
        "accuracy and sample quality; β = 1 is the best of the tested values for both. <b>Codebook size (Part A-2):</b> "
        "K sets the reconstruction ceiling (+0.6 dB per doubling) but not the FID, which is governed by the prior; no "
        "codebook collapse occurred. <b>WGAN vs WGAN-GP (Part B):</b> the only change is the Lipschitz mechanism and it "
        "reduces the FID by 46 %."))

    # 6. Conclusions
    s.append(P("6. Conclusions", h2))
    s.append(P(
        "(1) For reconstruction, VQ-VAE (about 25 dB) is clearly better than the β-VAE (16.8-20.7 dB), and in both "
        "families a smaller bottleneck (larger β, smaller K) costs PSNR. (2) For sampling quality, WGAN-GP (FID 63.2) "
        "&lt; VQ-VAE + PixelCNN (80-83) &lt; WGAN (116.2) &lt; β-VAE (141.8-277.6), with the same ordering in the "
        "perceptual comparison of the 100-sample grids. (3) For speed, the VAE is the cheapest to train and sample, the "
        "GANs are cheap to sample but the most expensive to train, and the VQ-VAE with an autoregressive prior is the "
        "slowest sampler. (4) No single model wins everything: the VAE family provides an encoder and a regular latent "
        "space, the VQ-VAE provides both good reconstruction and good samples at the price of slow sampling, and "
        "WGAN-GP provides the best and fastest samples but no encoder."))
    s.append(P(
        "<b>Limitations.</b> Each model was trained once (seed 42); the ± values describe sampling noise only, not "
        "training-seed variance, so differences of a few FID points (for example between the three codebook sizes) "
        "must not be read as an ordering. The budgets are moderate (25k GAN iterations, 40 epochs for the "
        "autoencoders) and the networks are small, so all absolute FIDs are higher than in published results with "
        "larger models; WGAN-GP in particular was still improving at the end of training. The 10,000-sample FID is "
        "biased slightly upwards with respect to the usual 50,000-sample FID, and the k-NN precision is sensitive to "
        "concentrated sample sets (see WGAN). Absolute times depend on the GPU (RTX 4090 here); their ratios are the "
        "more transferable result."))

    def footer(c, d):
        c.saveState(); c.setFont("DV", 7.5); c.setFillColor(colors.HexColor("#666666"))
        c.drawString(1.8 * cm, 1.1 * cm, "UDL Assignment 2 — Part C — Group 6")
        c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.7 * cm, title="UDL Assignment 2 - Part C", author="Group 6")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
