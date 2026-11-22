---
description: Conventions for Kubernetes manifests in this repo
applyTo: "**/*.{yaml,yml}"
---

# Kubernetes manifest rules

- Set `resources.requests` and `resources.limits` (cpu and memory) on every container.
- Pin every workload to a node with `nodeSelector: kubernetes.io/hostname: <node>` (nodes: `kevin-ubuntu`, `rogue1`, `stormtrooper`, `kevin-surface-book`).
- `rogue1` is tainted `db-only=true:NoSchedule`; add this toleration for anything scheduled there:
  ```yaml
  tolerations:
    - key: db-only
      operator: Equal
      value: "true"
      effect: NoSchedule
  ```
- Stateful data: use a local PV (`storageClassName: local-storage`, `Retain`, `nodeAffinity` on the same hostname as the pod) plus a PVC. Match the style of `k8s\StarCoder2\rogue1\pv.yaml`.
- Keep Deployment `metadata.name`, `selector.matchLabels.app`, and pod `labels.app` identical.
- Never put real passwords, tokens, or keys in YAML. Reference an existing Secret by name.
- Use a pinned image tag; avoid `:latest` for new files.
- Keep each resource kind in its own file (`deployment.yaml`, `service.yaml`, `pvc.yaml`, `pv.yaml`) unless the folder already uses a combined file.
- Validate with `kubectl apply --dry-run=server -f <file>` before applying. Deployment is manual `kubectl apply`, not ArgoCD.
- Never run `kubectl` commands yourself, including dry-runs and read-only ones. Ask the user to run them and paste the output.
