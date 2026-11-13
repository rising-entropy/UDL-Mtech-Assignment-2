# UDL Assignment 2 — (β-)VAE, VQ-VAE + PixelCNN, WGAN / WGAN-GP on CIFAR-10 (Group 6)

Course: Unsupervised Deep Learning (BITS WILP MTech). Problem statement: `Assignment2_Problem_Statement.pdf`.
This README is written so that a person or an agent picking up the folder can understand what exists,
how to re-run it (e.g. on the BITS Prayogshala GPU lab), and what must be updated after a re-run.

## 1. What the assignment requires (short)

| Part | Marks | Content |
|---|---|---|
| A-1 | 2 | β-VAE, β ∈ {1, 2, 4, 10}: PSNR vs. epoch, FID per β |
| A-2 | 3 | VQ-VAE with K ∈ {512, 256, 128}; extract discrete latents, train PixelCNN prior, sample, generate; FID, PSNR, codebook quality per K |
| B | 3 | WGAN (weight clipping) and WGAN-GP: train, sample, FID |
| C | 2 | 100 samples each for VAE (β=1), VQ-VAE (K=256), WGAN, WGAN-GP, perceptual comments, comparison of reconstruction error / sampling quality / speed |

Dataset CIFAR-10, pixels normalised to [0, 1]. Marks are deducted for poor or unexpected metrics, so numbers must be
real (not tuned) and show the well-known behaviour of each family.

**Deliverables:**
1. `udl_assignment2_partA_group6.ipynb` and `udl_assignment2_partB_group6.ipynb`, with all outputs embedded.
2. The matching `.html` exports.
3. `udl_assignment2_partC_group6.pdf` (Part C comparative discussion).
4. A `.jpg` screenshot showing the work was run on the BITS Prayogshala GPU infrastructure (**mandatory, not yet in this folder**).

Group submission: one upload for Group 6.

## 2. Folder layout

```
Assignment 2/
├── README.md                              this file
├── Assignment2_Problem_Statement.pdf
├── requirements.txt                       torch 2.5.1, torchvision 0.20.1, numpy, pandas, matplotlib, scipy, pillow, pytorch-fid
├── udl_assignment2_partA_group6.ipynb     Part A-1 (β-VAE) + A-2 (VQ-VAE + PixelCNN), executed, outputs embedded
├── udl_assignment2_partA_group6.html      nbconvert export of the above
├── udl_assignment2_partB_group6.ipynb     Part B (WGAN, WGAN-GP), executed
├── udl_assignment2_partB_group6.html
├── udl_assignment2_partC_group6.pdf       Part C report (7 pages, reportlab)
├── results/                               real-run outputs (committed, except the big .npz)
│   ├── partA_results.json                 metrics + per-epoch histories for vae_beta_{1,2,4,10}, vqvae_K_{512,256,128}
│   ├── partB_results.json                 metrics + histories for wgan, wgan_gp
│   ├── samples/                           vae_beta1_100.npy, vqvae_k256_100.npy, wgan_100.npy, wgan_gp_100.npy (100 samples each; Part C input)
│   ├── figures/                           every plot used in notebooks / Part C (partC_*.png were made by the Part C builder)
│   └── cifar10_inception_ref.npz          cached Inception stats of CIFAR-10 (197 MB, git-ignored, rebuilt automatically)
├── logs/                                  papermill logs of the real runs (git-ignored)
├── results_smoke/                         output of a UDL_SMOKE=1 dry run (git-ignored)
└── data/                                  CIFAR-10 download (git-ignored)
```

Git-ignored: `data/`, `results_smoke/`, `logs/`, `__pycache__/`, `results/cifar10_inception_ref.npz`.

## 3. How the notebooks work

Both notebooks are self-contained (no `.py` modules) and follow the same structure: prerequisites, imports,
configuration, CIFAR-10 load, helper functions, FID / precision / recall, models, training, plots, analysis.

* **Config cell** (cell 10 in both): `SMOKE = os.environ.get("UDL_SMOKE") == "1"` switches to a tiny dry run
  (5k images, 1 epoch, results go to `results_smoke/`). Otherwise the real config is used and results go to `results/`.
* **Seed** 42. `cudnn.benchmark = True`, so runs are not bit-reproducible, but metrics are stable (FID std over 3 rounds is 0.2-0.7).
* **Metrics:** PSNR on the 10k test set; FID with `pytorch-fid` Inception-v3 pool3 (2048-d) against the 50k-training-set
  statistics, 10,000 generated images, mean ± std over 3 sampling rounds; precision/recall (k=3 nearest-neighbour manifolds) on Inception features.
* `results/partA_results.json` / `partB_results.json` are deleted and rebuilt at the start of each run, so a run always overwrites them.

### Real-run configuration

| Model | Setup |
|---|---|
| β-VAE | conv encoder/decoder, latent 128, Adam lr 1e-3, batch 128, **40 epochs** per β; PSNR / KL / active units logged every epoch |
| VQ-VAE | dim 64, 8×8 latent grid, Adam lr 5e-4, **40 epochs** per K |
| PixelCNN prior | on the 8×8 code maps, Adam lr 1e-3, **40 epochs** per K; ancestral sampling then VQ-VAE decoder |
| WGAN | weight clipping 0.01, n_critic 5, batch 64, **25,000 generator iterations**, FID checkpoint every 2,500 |
| WGAN-GP | gradient penalty λ = 10, same generator/critic and schedule |

### Cell contents that matter

* Markdown cells tagged `analysis_vae`, `analysis_vq`, `analysis_gan` hold the written analysis with **hard-coded numbers**
  from the real run (tables and prose). They are not generated from the JSON.
* "Expected Result" markdown cells (Part A cells 6 and 36, Part B cell 6) are the pre-run expectations.
* The last cell in each notebook ("Verification Screenshots") says the Prayogshala `.jpg` is submitted separately.

## 4. Results of the existing local run (RTX 3080 Laptop, 8 GB, torch 2.5.1+cu121)

| Model | Test PSNR (dB) | FID ↓ | Precision | Recall | Train time |
|---|---|---|---|---|---|
| β-VAE β=1 | 20.71 | 141.5 | 0.264 | 0.002 | 366 s |
| β-VAE β=2 | 19.45 | 172.8 | 0.162 | 0.000 | 372 s |
| β-VAE β=4 | 18.24 | 214.6 | 0.090 | 0.000 | 386 s |
| β-VAE β=10 | 16.84 | 275.5 | 0.059 | 0.000 | 391 s |
| VQ-VAE + PixelCNN K=512 | 25.38 | 78.4 | 0.537 | 0.116 | 793 s + 620 s |
| VQ-VAE + PixelCNN K=256 | 24.86 | 80.4 | 0.555 | 0.107 | 751 s + 551 s |
| VQ-VAE + PixelCNN K=128 | 24.29 | 79.0 | 0.559 | 0.099 | 739 s + 519 s |
| WGAN | n/a | 110.3 | 0.663 | 0.002 | 1251 s |
| WGAN-GP | n/a | 64.2 | 0.580 | 0.137 | 1774 s |

Expected story, which the results support: PSNR falls and FID rises with β; VQ-VAE reconstructs much better than the VAE
and its PixelCNN samples are far sharper; K sets the reconstruction ceiling (about +0.5 dB per doubling) while FID is governed by the prior
and all codebook entries stay in use (no collapse); WGAN-GP beats WGAN-clip clearly on FID and recall. Overall FID order:
WGAN-GP < VQ-VAE+PixelCNN < WGAN < β-VAE.

Expected total wall time on a similar GPU: Part A about 1.5-2 h, Part B about 50 min (plus a few minutes of Inception / FID).

## 5. How to run (BITS Prayogshala or any CUDA machine)

### 5.1 Environment

Python 3.10+, a CUDA build of PyTorch. The local run used torch 2.5.1+cu121 / torchvision 0.20.1 / numpy 2.2 / pytorch-fid 0.3.0.

```bash
python -m venv udl_a2_venv && source udl_a2_venv/bin/activate     # on Windows: udl_a2_venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib scipy pillow pytorch-fid jupyter papermill nbconvert
```

(Adjust the CUDA wheel index to whatever the lab driver supports. If torch is already provided on the lab, only install the rest.)
Check: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

### 5.2 Network downloads (two things are fetched on first use)

1. **CIFAR-10** via `torchvision.datasets.CIFAR10(download=True)` into `./data` (170 MB).
2. **Inception weights** (`pt_inception-2015-12-05-6726825d.pth`, ~95 MB) via `pytorch-fid` into `~/.cache/torch/hub/checkpoints/`.

On the original laptop Python's SSL could not verify certificates, so both files were fetched manually with `curl.exe`
(CIFAR tar into `data/`, extracted so that `data/cifar-10-batches-py/` exists; Inception weights into the checkpoints folder).
If downloads fail on the lab machine, do the same:

```bash
mkdir -p data && curl -L -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && tar -xzf data/cifar-10-python.tar.gz -C data
mkdir -p ~/.cache/torch/hub/checkpoints && curl -L -o ~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
```

### 5.3 Dry run first (about a minute per notebook)

```bash
UDL_SMOKE=1 papermill udl_assignment2_partA_group6.ipynb /tmp/smokeA.ipynb --log-output
UDL_SMOKE=1 papermill udl_assignment2_partB_group6.ipynb /tmp/smokeB.ipynb --log-output
```

On PowerShell use `$env:UDL_SMOKE="1"` before the command. Smoke runs write to `results_smoke/` and do not touch `results/`.
Write the smoke notebooks to a different path as above so the real notebooks keep their outputs.

### 5.4 Real run

Run Part A and Part B **one at a time**. On the 8 GB laptop two concurrent training jobs thrashed memory and stalled both.
FID's 2048×2048 matrix square root is CPU-heavy, so do not expect GPU-only speed there.

```bash
mkdir -p logs
papermill udl_assignment2_partA_group6.ipynb udl_assignment2_partA_group6.ipynb --log-output 2>&1 | tee logs/partA.log
papermill udl_assignment2_partB_group6.ipynb udl_assignment2_partB_group6.ipynb --log-output 2>&1 | tee logs/partB.log
jupyter nbconvert --to html udl_assignment2_partA_group6.ipynb
jupyter nbconvert --to html udl_assignment2_partB_group6.ipynb
```

Alternatively open both notebooks in JupyterLab on the lab server and use "Restart kernel and run all". Either way the saved
`.ipynb` must contain the outputs (the assignment requires embedded outputs), then export the HTML.

Part B does not depend on Part A, but both use `results/cifar10_inception_ref.npz` (built by whichever notebook runs first).

## 6. IMPORTANT: what a re-run on BITS infra invalidates

A fresh run produces slightly different numbers (seed and cudnn nondeterminism). Therefore, if you re-execute on Prayogshala:

1. The `analysis_*` markdown cells and the "Part A summary" text contain hard-coded numbers from the local run. Read the new
   `results/partA_results.json` and `partB_results.json` and update those cells so prose and tables match the new outputs
   (rerun `nbconvert` afterwards). Do not change the training config to chase numbers; keep results honest. If a trend changes
   (e.g. FID ordering of the K values), say so in the analysis.
2. The **Part C PDF is stale** after a re-run. It was built with `reportlab` from `results/*.json`, `results/samples/*.npy` and the
   figures `results/figures/partC_*.png`, by `tools/build_partC.py` (figures and grids by `tools/partC_figures.py`; run both from the repo root).
   The scores table is read from the JSON files, but the prose quotes numbers by hand, so revise it after a re-run. The report contains: 100-sample grids (10×10, upscaled) for VAE β=1, VQ-VAE K=256, WGAN, WGAN-GP and the real
   test images; a bar chart of FID and precision/recall (`partC_fid_pr.png`); a speed chart (`partC_speed.png`); the group table;
   and sections: 1 setup/protocol, 2 evaluation scores, 3 100 samples + perceptual comments, 4 comparative discussion
   (reconstruction error, sampling quality, speed), 5 ablation highlights. Use the existing PDF as the template for wording and layout.
3. If the old results are good enough, the lower-risk option is to **keep the existing notebooks / HTML / PDF** and only add the
   Prayogshala screenshot, but the assignment says the work must have been run on that infrastructure, so decide with the group.
   Do not present local results as lab results in any text.

## 7. Remaining to do (as of this README)

* [ ] Run (or at least run and screenshot) on BITS Prayogshala and save a `.jpg` showing the lab environment (hostname / GPU / the notebook running).
* [ ] If re-running: refresh analysis cells, regenerate HTML, rebuild the Part C PDF as described in section 6.
* [ ] Final check before upload: both `.ipynb` files have embedded outputs and no error cells; HTML files are newer than the notebooks; PDF numbers match the notebook numbers; group table is the same everywhere.

## 8. Group 6

| Name | BITS ID | Contribution % |
|---|---|---|
| Brincita Prima Cutinha | 2025AG05256 | 100 |
| Devang Ajay Kamble | 2025AE05173 | 100 |
| Saksham Jindal | 2025AE05983 | 100 |
| Shantanu Biswas | 2025AE05079 | 100 |
| Ashish Chauhan | 2025AG05928 | 100 |
