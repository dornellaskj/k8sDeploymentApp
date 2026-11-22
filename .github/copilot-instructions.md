# Copilot instructions for k8sDeploymentApp

Homelab GitOps-style monorepo for a local 4-node MicroK8s cluster (Kubernetes v1.30). It holds raw YAML manifests, GPU/LLM workloads, and the GloboTicket .NET demo (`catalog`, `ordering`, `frontend`). There is no Helm/Kustomize for apps (ArgoCD and Harbor are the exceptions) and no CI that applies manifests.

## Deployment workflow
- **Never run `kubectl` (or `microk8s kubectl`, `helm`, or any other command that talks to the cluster) yourself.** This applies to read-only commands too (`get`, `describe`, `logs`, `--dry-run`). Instead, give the user the exact command in a code block, ask them to run it, and wait for them to paste the output before continuing.
- Manifests are applied manually with `kubectl apply -f <file-or-dir>`. ArgoCD is installed (`k8s\argocd`) but is **not** how workloads are deployed.
- Always suggest `kubectl apply --dry-run=server -f <path>` (or `--dry-run=client` offline) before a real apply, and `kubectl rollout status` afterwards.
- Never run destructive commands (`kubectl delete`, PV/PVC removal, node drain) without the user asking. PVs use `Retain`; keep it that way.

## Cluster topology
| Node | Hardware | Role / notes |
|---|---|---|
| `kevin-ubuntu` | 2x RTX 5060 Ti (16 GB each, 32 GB total), 16 CPU, ~32 GB RAM, no taints | Primary LLM inference (Ollama). Also where `kubectl` is run. |
| `rogue1` | 2x RTX 3060 (12 GB each, 24 GB total), 40 CPU, ~115 GB RAM | Labeled `db-node=true`, tainted `db-only=true:NoSchedule`. Any pod placed here needs a matching toleration. |
| `stormtrooper` | 1x RTX 2070 (8 GB), 8 CPU, ~16 GB RAM, no taints | Runs embedding models (currently `ollama-rogue1`, 4Gi request / 8Gi limit). Also hosts the registry, metabase, cert-manager. |
| `kevin-surface-book` | No GPU, 4 CPU, ~8 GB RAM, no taints, Ubuntu 22.04 | General-purpose, small workloads only (currently elasticsearch-webui, jiradata-api, quantizer redis). Keep memory requests small. |

- Pin workloads with `nodeSelector: kubernetes.io/hostname: <node>`; the existing manifests all do this.
- Local PVs use `storageClassName: local-storage`, `persistentVolumeReclaimPolicy: Retain`, a `local.path`, and a `nodeAffinity` on `kubernetes.io/hostname` matching the pod's node.
- NVIDIA GPU operator is installed; request GPUs with `nvidia.com/gpu` in `resources.limits`. GPU time-slicing is off (`gpu.replicas=1`), so one GPU = one pod.
- Networking: MetalLB (`metallb-config.yaml`) plus Traefik ingress (`k8s\traefik`).

## Repo layout
- `k8s\<app>\` - infrastructure and platform manifests (argocd, harbor, grafana, mongo, neo4j, Postgres, sqlServer, traefik, vsCode, StarCoder2/Ollama).
- `GPU\`, `llm\`, `llama_factory\`, `quantizer\`, `CPU-AI-Orchestrator\` - AI workloads.
- `catalog\`, `ordering\`, `frontend\`, `chatbot\`, `whoami\` - application code and their manifests.
- `mysql\`, `metabase\`, `minecraft\`, `kube-bench\`, `defectdojo\`, `codescene\` - other services and tooling.

## Gotchas
- `k8s\StarCoder2\rogue1\` is misleadingly named: it deploys Ollama to **`stormtrooper`** (PV `ollama-pv-rogue1`, path `/ollama1`). Verify the real `nodeSelector` rather than trusting folder or resource names.
- Secret manifests (`*-secret*.yaml`, `mongodb-secrets.yaml`) already exist in the repo. Do not add new plaintext credentials; prefer `kubectl create secret` and reference by name.
- Prefer pinned image tags over `:latest` for new manifests.
