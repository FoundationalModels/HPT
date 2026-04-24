# Running HPT on Tufts HPC

This guide is for lab members who want to launch HPT training jobs on the Tufts HPC cluster. The environment and repo are already set up — you just need to log in and submit the job.

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

## 4. Submit the Job

The SLURM script `hpc.sh` is already in the repo root. Submit it with:

```bash
sbatch hpc.sh
```

SLURM will print a job ID, e.g.:

```
Submitted batch job 36582500
```

Keep this ID — you'll use it to monitor the job.

The script requests 1 A100 GPU on the `gpu,preempt` partition and runs HPT inside the shared Singularity container image:

```bash
/cluster/tufts/hrilab/hlu07/hpt.sif
```

The command executed inside the container is:

```bash
python -m hpt.run
```

> **Note:** The `preempt` partition means your job may be interrupted if higher-priority jobs need the GPU. If this happens, resubmit with `sbatch hpc.sh`.

---

## 5. Monitor the Job

```bash
# One-time status check
squeue -u {your_username}

# Live view, refreshes every 2 seconds (Ctrl+C to stop)
watch squeue -u {your_username}
```

The `ST` column shows the status: `PD` = waiting for a GPU, `R` = running. Log files are created once the job starts running.

```bash
# Stream live output (replace 36582500 with your job ID)
tail -f /cluster/tufts/hrilab/hlu07/HPT/logs/hpt_36582500.out

# Check errors if something goes wrong
cat /cluster/tufts/hrilab/hlu07/HPT/logs/hpt_36582500.err

# Check CPU/memory efficiency after the job finishes
seff 36582500

# Cancel a job if needed
scancel 36582500
```

---

## 6. Output

Training output, checkpoints, and figures are saved to the `output/` directory inside the HPT repo:

```
/cluster/tufts/hrilab/hlu07/HPT/output/
```

Training is also tracked via Weights & Biases (wandb) if configured.

---

## 7. Download Results to Your Local Machine

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
| Submit job | `sbatch hpc.sh` |
| Check queue | `squeue -u {your_username}` |
| View live logs | `tail -f logs/hpt_<job_id>.out` |
| View errors | `cat logs/hpt_<job_id>.err` |
| Cancel job | `scancel <job_id>` |

---

## Support

For HPC issues, email: `tts-research@tufts.edu`  
HPC documentation: [rtguides.it.tufts.edu/hpc](https://rtguides.it.tufts.edu/hpc)



cat > /cluster/tufts/hrilab/hlu07/test_hpt.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=hpt_test
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_test_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p preempt
#SBATCH --gres=gpu:1

module load singularity/3.8.4
export WANDB_API_KEY=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

singularity exec --nv \
  --bind /cluster/tufts/hrilab/hlu07/HPT:/workspace \
  /cluster/tufts/hrilab/hlu07/hpt.sif \
  bash -c "cd /workspace && python -m hpt.run +mode=debug"
EOF