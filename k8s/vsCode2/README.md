# Second Remote VS Code Dev Pod (GPU #2)

A second dev pod on `rogue1`, identical to [../vsCode](../vsCode) but with its own
GPU, workspace volume and SSH Service, so training and inference can run side by side.

It reuses the `vscode-remote-config` ConfigMap and `vscode-remote-ssh-key` Secret
from `../vsCode`, so apply that folder's `configmap.yaml` and create the secret first.

| Resource | Name |
|---|---|
| Deployment / Service | `vscode-remote-2` |
| PV / PVC | `vscode-workspace-pv-rogue1-2` / `vscode-workspace-pvc-rogue1-2` |
| Host path | `/mnt/backup/vscode-workspace-2` (200Gi) |
| GPU | `nvidia.com/gpu: 1` |

## Deploy

Create the host directory on `rogue1` once:

```bash
sudo mkdir -p /mnt/backup/vscode-workspace-2
```

```powershell
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get svc vscode-remote-2
```

Add a second `Host` entry to `~/.ssh/config` using the new `EXTERNAL-IP`.

## GPU capacity

`rogue1` has 2 GPUs. Pods in `../vsCode` (1) and this folder (1) use both, so the
Ollama deployments in `../StarCoder2` (1 or 2 GPUs) will stay `Pending` unless one of
the dev pods is scaled down. Verify with `kubectl describe node rogue1`
(`nvidia.com/gpu` under Allocatable and Allocated resources).
