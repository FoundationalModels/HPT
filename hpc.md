# Running HPT on Tufts HPC

This guide is for lab members who want to run HPT data downloads and pretraining jobs on the Tufts HPC cluster.

---

## 1. Prerequisites

### HPC Account

If you don't have a Tufts HPC account yet:

1. Go to [it.tufts.edu](https://it.tufts.edu) and search for "HPC Account Request"
2. Fill out the request form with your Tufts credentials and PI/lab information
3. Wait for confirmation from `tts-research@tufts.edu`

### Tufts VPN

If you are off-campus, connect to the Tufts VPN before accessing the cluster:

1. Go to [vpn.tufts.edu](https://vpn.tufts.edu) and download the Cisco AnyConnect client
2. Connect to `vpn.tufts.edu` using your Tufts UTLN and password

---

## 2. Log In

### Option A: SSH (Local Terminal)

```bash
ssh {your_username}@login.pax.tufts.edu
```

### Option B: OnDemand Web Portal

1. Go to [ondemand.pax.tufts.edu](https://ondemand.pax.tufts.edu)
2. Log in with your Tufts credentials
3. Open a terminal via **Clusters → Tufts HPC Shell Access**

---

## 3. Navigate to the HPT Repo

```bash
cd /cluster/tufts/hrilab/hlu07/HPT
```

---

## 4. Updating the Container Image

The Singularity image lives at `/cluster/tufts/hrilab/hlu07/hpt.sif`.  When the
Docker image on Docker Hub is updated, pull the new image to replace it:

```bash
# Log in to a GPU node or use an interactive session — pulling on a login node is slow
module load singularity/3.8.4
singularity pull --force \
  /cluster/tufts/hrilab/hlu07/hpt.sif \
  docker://helenlu66/hpt:latest
```

The Docker image is built and pushed automatically from the repo root:

```bash
# Run locally before pushing
docker build -t helenlu66/hpt:latest -t helenlu66/hpt:<git-hash> .
docker push helenlu66/hpt:latest
docker push helenlu66/hpt:<git-hash>
```

---

## 5. Download Simulation Data

Before pretraining, download and preprocess all domain datasets onto the cluster.
This is a one-time step (the data persists across jobs).

```bash
cd /cluster/tufts/hrilab/hlu07/HPT
sbatch hpc_download_sim_data.sh
```

What it does:
- Downloads Robomimic, Adroit, Arnold, and ManiSkill datasets
- Generates Drake demonstrations via Fleet-Tools
- Generates PyBullet Trifinger episodes online

Monitor the job:

```bash
squeue -u {your_username}
tail -f logs/hpt_download_sim_data_<job_id>.out
```

---

## 6. Launch Pretraining

Once the data download job has completed, submit the pretraining job:

```bash
cd /cluster/tufts/hrilab/hlu07/HPT
sbatch hpc_pretrain.sh
```

The WandB API key is hardcoded inside `hpc_pretrain.sh` so the job runs fully unattended.

The script:
- Requests 1 A100 GPU, 8 CPUs, 128 GB RAM, 72-hour time limit
- Runs on the `gpu,preempt` partition (may be interrupted by higher-priority jobs)
- Automatically resumes from the last checkpoint if preempted — just re-submit with `sbatch hpc_pretrain.sh`
- Builds ResNet-precomputed zarr caches for each domain on the first run; subsequent runs (including after preemption) load the cached zarrs directly

### Overriding defaults

You can tune any parameter via environment variables before submitting:

```bash
# Fewer episodes per domain for a faster run
EPISODE_CNT=1000 sbatch hpc_pretrain.sh

# Specific subset of domains
DOMAINS=mujoco_robomimic,mujoco_adroit sbatch hpc_pretrain.sh
```

### Smoke test first

Before launching a full 72-hour run, validate the entire pipeline with a short smoke test:

```bash
sbatch hpc_pretrain_smoke_test_all_data.sh
```

This runs 25 episodes per domain and 10 training iterations (~1 hour).

---

## 7. Monitor the Job

```bash
# One-time status check
squeue -u {your_username}

# Live view, refreshes every 2 seconds (Ctrl+C to stop)
watch squeue -u {your_username}
```

The `ST` column shows the status: `PD` = waiting for a GPU, `R` = running.

```bash
# Stream live output (replace <job_id> with yours)
tail -f /cluster/tufts/hrilab/hlu07/HPT/logs/hpt_pretrain_<job_id>.out

# Check errors if something goes wrong
cat /cluster/tufts/hrilab/hlu07/HPT/logs/hpt_pretrain_<job_id>.err

# Check CPU/memory efficiency after the job finishes
seff <job_id>

# Cancel a job
scancel <job_id>
```

Training loss and validation metrics are logged to Weights & Biases under the
project `hpt-pretrain`, tag `hpc_pretrain_all_data`.

---

## 8. Output

Checkpoints and config are saved under `output/` inside the repo:

```
/cluster/tufts/hrilab/hlu07/HPT/output/<date>_hpc_pretrain_all_data/
  model.pth            — latest full model weights
  trunk.pth            — trunk-only weights (for fine-tuning)
  training_state.pth   — full optimizer + scheduler state (for resuming)
  training_state_<N>.pth — periodic snapshots every 10 000 steps
  config.yaml          — hydra config snapshot
```

---

## 9. Download Results to Your Local Machine

From your **local terminal**:

```bash
rsync -avz hlu07@login.pax.tufts.edu:/cluster/tufts/hrilab/hlu07/HPT/output/ ./hpt_output/
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Connect (SSH) | `ssh {your_username}@login.pax.tufts.edu` |
| Connect (Web) | [ondemand.pax.tufts.edu](https://ondemand.pax.tufts.edu) |
| Go to project | `cd /cluster/tufts/hrilab/hlu07/HPT` |
| Update container | `singularity pull --force hpt.sif docker://helenlu66/hpt:latest` |
| Download data | `sbatch hpc_download_sim_data.sh` |
| Smoke test | `sbatch hpc_pretrain_smoke_test_all_data.sh` |
| Launch pretraining | `export WANDB_API_KEY=<key> && sbatch hpc_pretrain.sh` |
| Check queue | `squeue -u {your_username}` |
| View live logs | `tail -f logs/hpt_pretrain_<job_id>.out` |
| View errors | `cat logs/hpt_pretrain_<job_id>.err` |
| Cancel job | `scancel <job_id>` |
| Download results | `rsync -avz hlu07@login.pax.tufts.edu:.../output/ ./hpt_output/` |

---

## Support

For HPC issues, email: `tts-research@tufts.edu`  
HPC documentation: [rtguides.it.tufts.edu/hpc](https://rtguides.it.tufts.edu/hpc)
