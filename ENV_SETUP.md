# Environment Setup (verified run book)

Steps used to get this repo running on the RunPod GPU box (image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
Python 3.12, RTX 4090, CUDA 12.8). Both notebooks passed the `UDL_SMOKE=1` dry run with this setup.
If your environment matches, follow these steps and **skip the smoke test**.

## 1. Clone to the workspace root

```bash
cd /workspace
git clone https://github.com/rising-entropy/UDL-Mtech-Assignment-2 repo_tmp
shopt -s dotglob && mv repo_tmp/* . && rmdir repo_tmp
```

The box already had a `requirements.txt` from the Docker image, so it was renamed to `requirements.docker.txt` first so the repo's copy wouldn't overwrite it.

## 2. Python packages

**Do not** `pip install -r requirements.txt` on this image. That file pins `torch==2.5.1 / torchvision==0.20.1`, which
would downgrade the preinstalled CUDA 12.8 build (`torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`).
The notebooks run fine on torch 2.8. Install only these:

```bash
pip install pytorch-fid pandas pillow papermill "scipy<1.16"
```

**`scipy<1.16` is required.** The image ships scipy 1.18, which removed the `disp` argument of
`scipy.linalg.sqrtm`. `pytorch-fid` 0.3.0 still passes it, so every FID computation fails with
`TypeError: sqrtm() got an unexpected keyword argument 'disp'`. scipy 1.15.3 works with numpy 2.1.2.

Already present in the image: torch, torchvision, numpy 2.1.2, matplotlib, scikit-learn, jupyter/jupyterlab, ipykernel, tqdm.

**Slow PyPI:** files.pythonhosted.org ran at ~56 kB/s here. The Aliyun mirror was about 6× faster:
`pip install -i https://mirrors.aliyun.com/pypi/simple/ "scipy<1.16"`. pip still checks the hashes, but if
you download a wheel by hand, compare its sha256 with `https://pypi.org/pypi/<pkg>/<ver>/json` before installing.

## 3. Dataset (CIFAR-10) and FID assets

The official CIFAR-10 server (cs.toronto.edu) ran at ~25 kB/s here, about an hour for 170 MB. **Use the copy in the repo instead:**

```bash
cd /workspace/data
cat cifar-10-python.tar.gz.part-* > cifar-10-python.tar.gz
md5sum cifar-10-python.tar.gz   # must be c58f30108f718f92721af3b95e74349a
```

The archive is split into two parts because GitHub rejects files over 100 MB. With the rejoined `.tar.gz` in `data/`, torchvision
checks its MD5, extracts it and skips the download (`download=True` in the notebook is fine).

Where the repo copy came from: `https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python` (~15 MB/s).
It is byte-identical to the official archive: size 170,498,071 bytes, MD5 `c58f30108f718f92721af3b95e74349a` (the same checksum torchvision checks).
Both notebooks note this in the "Load the dataset" cell.

Other assets, all cached automatically on the first run:

| Asset | Size | Cache location |
|---|---|---|
| FID Inception-v3 weights | 91 MB | `~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth`. Prefetch: `python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3])"` (took ~6 min here) |
| CIFAR-10 Inception reference stats | 197 MB | `results/cifar10_inception_ref.npz` (git-ignored; computed locally on the GPU) |

## 4. Run

```bash
cd /workspace
mkdir -p logs
papermill udl_assignment2_partA_group6.ipynb udl_assignment2_partA_group6.ipynb --log-output 2>&1 | tee logs/partA.log
papermill udl_assignment2_partB_group6.ipynb udl_assignment2_partB_group6.ipynb --log-output 2>&1 | tee logs/partB.log
jupyter nbconvert --to html udl_assignment2_partA_group6.ipynb udl_assignment2_partB_group6.ipynb
```

Or open the notebooks in JupyterLab and Run All (make sure `UDL_SMOKE` is **not** set).

## 5. Smoke test (only if the environment changed)

```bash
mkdir -p /tmp/smoke
UDL_SMOKE=1 papermill udl_assignment2_partA_group6.ipynb /tmp/smoke/smokeA.ipynb --log-output
UDL_SMOKE=1 papermill udl_assignment2_partB_group6.ipynb /tmp/smoke/smokeB.ipynb --log-output
```

Output goes to `results_smoke/` (git-ignored). papermill fails if the output notebook's folder doesn't exist, so create it first.

## Gotchas found during setup

- Don't downgrade torch (see step 2).
- Pin `scipy<1.16`, otherwise FID crashes (see step 2).
- Rejoin the CIFAR parts before the first run, otherwise the notebook spends about an hour downloading.
- papermill fails with `FileNotFoundError: output folder ... doesn't exist` if the output directory hasn't been created.
