# Remote VS Code Dev Pod (GPU-enabled)

This folder deploys a Kubernetes pod you can connect to from your local **VS Code
Desktop** app to run GPU-accelerated development workloads. The pod runs
`sshd` on a CUDA-enabled Ubuntu image, backed by:

- 2-4 CPUs / 24Gi RAM
- 1 NVIDIA GPU (`nvidia.com/gpu: 1` limit — requires the
  [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) installed
  on the cluster)
- A persistent `/home/coder` volume (`vscode-workspace-pvc-rogue1`) so your entire
  home directory - files anywhere in it, not just one subfolder - survives pod
  restarts
- A `LoadBalancer` service (via MetalLB, see `metallb-config.yaml`) exposing
  SSH on port 22 with an external IP

Authentication is **SSH public key only** — password and root login are
disabled in `sshd_config` for security.

## Files

| File | Purpose |
|---|---|
| `configmap.yaml` | `sshd_config` + the container entrypoint script that installs `sshd`/dev tools and starts the SSH daemon |
| `pv.yaml` | Local persistent volume pinned to `rogue1` |
| `pvc.yaml` | Persistent volume claim for `/home/coder` |
| `deployment.yaml` | The pod spec (image, resources, GPU, volumes, node selector) |
| `service.yaml` | `LoadBalancer` service exposing port 22 |

## 1. Generate an SSH key pair (if you don't already have one)

```powershell
ssh-keygen -t ed25519 -C "vscode-remote" -f $HOME\.ssh\vscode-remote
```

This creates `vscode-remote` (private key) and `vscode-remote.pub` (public key).

## 2. Create the secret holding your public key

The public key is **not** checked into this repo — create the secret directly
from your local `.pub` file:

```powershell
kubectl create secret generic vscode-remote-ssh-key `
  --from-file=authorized_keys=$HOME\.ssh\vscode-remote.pub
```

## 3. Update the node selector

`deployment.yaml` pins the pod to `kubernetes.io/hostname: rogue1`.
Change this to the hostname of whichever node in your cluster has the NVIDIA
GPU and device plugin installed:

```powershell
kubectl get nodes -o wide
```

## 4. Deploy

Create the local volume directory on `rogue1` once:

```bash
sudo mkdir -p /mnt/dbhot/vscode-workspace
```

```powershell
kubectl apply -f configmap.yaml
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## 5. Get the external IP

```powershell
kubectl get svc vscode-remote
```

Wait for `EXTERNAL-IP` to populate (MetalLB assigns it from the pool in
`metallb-config.yaml`).

## 6. Install VS Code plugins locally

Install these extensions in your local VS Code Desktop:

- **Remote - SSH** (`ms-vscode-remote.remote-ssh`) — required, lets VS Code
  connect to and run entirely inside the remote pod
- **Remote Explorer** (`ms-vscode.remote-explorer`) — optional, adds a UI panel
  for managing SSH targets

## 7. Add an SSH config entry

Add an entry to `~/.ssh/config` (create the file if it doesn't exist):

```
Host vscode-remote-k8s
    HostName <EXTERNAL-IP-FROM-STEP-5>
    User coder
    IdentityFile ~/.ssh/vscode-remote
    StrictHostKeyChecking accept-new
```

## 8. Connect

1. Open the Command Palette (`Ctrl+Shift+P`)
2. Run **Remote-SSH: Connect to Host...**
3. Select `vscode-remote-k8s`
4. Once connected, open the folder `/home/coder` (or any subfolder under it) -
   everything under your home directory now persists across pod restarts

You're now running VS Code entirely on the remote pod — the integrated
terminal, debuggers, and any extensions you install there execute on the
GPU-backed node.

## 9. Verify GPU access

In the remote VS Code integrated terminal:

```bash
nvidia-smi
```

This should list the GPU allocated to the pod.

## Notes

- The previous `vscode-workspace-pvc` claim is retained because its host-local
  volume is bound to `stormtrooper`. The Rogue claim starts with an empty
  workspace unless data is migrated separately.
- To change SSH keys, rotate the secret:
  `kubectl create secret generic vscode-remote-ssh-key --from-file=authorized_keys=... --dry-run=client -o yaml | kubectl replace -f -`
  then restart the pod (`kubectl rollout restart deployment/vscode-remote`).
- The container installs dev tools (`git`, `build-essential`, `python3`) on
  first start via `entrypoint.sh`. For faster startups, consider baking a
  custom image with these pre-installed instead of installing at runtime.
