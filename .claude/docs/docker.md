# Docker Container

## Container

| Property | Value |
|---|---|
| Name | `nas-for-moe` |
| Image | `nas-for-moe:0.0` |
| Status | Running (detached) |
| GPU | RTX 3080 Ti via `--gpus all --runtime=nvidia` |
| Port | `8887` → `8887` (JupyterLab) |
| Shared memory | 64 GB (`--shm-size=64g`) |
| IPC | host |

## Volume Mounts

| Host | Container |
|---|---|
| `/home/petr` | `/pbabkin` |

Project is at `/pbabkin/main/mipt/nas-for-moe/` inside the container.

## Running Commands

**Before running any command**, check if the container is running. If it is not running, start it first:

```bash
docker start nas-for-moe
```

All experiments and Python scripts **must** be executed inside the container:

```bash
docker exec nas-for-moe <command>

# Example: run a Python script
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/count_params.py

# Example: launch JupyterLab
docker exec -d nas-for-moe /app/launch_jupyter.sh
```

> **Important:** If `docker exec` fails with "container is not running", run `docker start nas-for-moe` and retry.

## Starting / Stopping

The container runs `sleep infinity` as its entrypoint, so it stays alive.

```bash
# Start if stopped
docker start nas-for-moe

# Stop
docker stop nas-for-moe

# Remove
docker rm -f nas-for-moe
```

## Rebuild Image

```bash
cd /home/petr/main/mipt/nas-for-moe/docker && docker build -t nas-for-moe:0.0 . -f Dockerfile
```

## Stack

- Base: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- Python 3.10
- PyTorch 2.5.1 + CUDA 12.4
- JupyterLab 4.2.7 on port 8887
