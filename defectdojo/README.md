# DefectDojo — Kubernetes Deployment

DefectDojo is a DevSecOps vulnerability management platform. This folder contains manifests for deploying the open-source edition to Kubernetes.

## Architecture

| Component | Manifest | Description |
|---|---|---|
| Namespace | `namespace.yaml` | Isolated `defectdojo` namespace |
| Secrets | `secrets.yaml` | DB URL, Django secret key, AES key, admin password |
| Valkey | `valkey-deployment.yaml` / `valkey-service.yaml` | Redis-compatible message broker for Celery |
| Media volume | `media-pvc.yaml` | Persistent volume for uploaded files |
| Initializer | `initializer-job.yaml` | Runs DB migrations and creates the admin user (one-time Job) |
| App | `deployment.yaml` | nginx + uWSGI (Django) as a two-container pod |
| Service | `service.yaml` | LoadBalancer exposing the app on port 80 |
| Celery Worker | `celery-worker-deployment.yaml` | Async task processing (deduplication, JIRA sync) |
| Celery Beat | `celery-beat-deployment.yaml` | Scheduled tasks (SLA alerts, engagement reminders) |

> PostgreSQL is provided by an existing cluster database at `192.168.86.159:5432`.

---

## Prerequisites

- `kubectl` configured against your target cluster
- A PostgreSQL database reachable at `192.168.86.159:5432` with a database and user created for DefectDojo
- A storage class that supports `ReadWriteMany` for the media PVC (e.g. NFS, Longhorn, CephFS). If only `ReadWriteOnce` is available, change `media-pvc.yaml` and ensure nginx and uWSGI remain co-located in the same pod (already the case in `deployment.yaml`).

### Create the PostgreSQL database and user

```sql
CREATE DATABASE defectdojo;
CREATE USER defectdojo WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE defectdojo TO defectdojo;
```

---

## Configuration

Before deploying, update **`secrets.yaml`** with real values:

| Key | How to generate |
|---|---|
| `DD_SECRET_KEY` | `openssl rand -base64 50` |
| `DD_CREDENTIAL_AES_256_KEY` | `openssl rand -base64 32` |
| `DD_DATABASE_URL` | `postgresql://<user>:<pass>@192.168.86.159:5432/<db>` |
| `POSTGRES_PASSWORD` | Must match the password in `DD_DATABASE_URL` |
| `DD_ADMIN_PASSWORD` | Initial admin account password |

> **Important:** `DD_CREDENTIAL_AES_256_KEY` encrypts stored API keys (SonarQube, Jira, etc.). Changing it after the first deployment will break decryption of any saved credentials.

---

## Deployment

Apply the manifests in the order below. Each step depends on the previous one completing successfully.

### 1. Namespace and secrets

```bash
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
```

### 2. Message broker

```bash
kubectl apply -f valkey-deployment.yaml
kubectl apply -f valkey-service.yaml
```

### 3. Media volume

```bash
kubectl apply -f media-pvc.yaml
```

### 4. Run the initializer (migrations + admin user)

```bash
kubectl apply -f initializer-job.yaml

# Wait for the Job to complete before continuing
kubectl wait --for=condition=complete job/defectdojo-initializer \
  -n defectdojo --timeout=300s
```

### 5. Deploy the application

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 6. Deploy Celery workers

```bash
kubectl apply -f celery-worker-deployment.yaml
kubectl apply -f celery-beat-deployment.yaml
```

### Apply everything at once (after first-time setup)

```bash
kubectl apply -f .
```

---

## Verify

```bash
# Check all pods are Running
kubectl get pods -n defectdojo

# Get the external IP assigned to the LoadBalancer
kubectl get svc defectdojo -n defectdojo

# Tail application logs
kubectl logs -n defectdojo -l app=defectdojo -c uwsgi -f
```

Access DefectDojo in your browser at the LoadBalancer IP on port 80.  
Default credentials: **admin** / value of `DD_ADMIN_PASSWORD` in `secrets.yaml`.

---

## Upgrades

Re-run the initializer Job after each version bump to apply new migrations:

```bash
# Delete the completed job first, then re-apply with the new image tag
kubectl delete job defectdojo-initializer -n defectdojo
kubectl apply -f initializer-job.yaml

kubectl wait --for=condition=complete job/defectdojo-initializer \
  -n defectdojo --timeout=300s

# Then restart the app pods to pick up the new image
kubectl rollout restart deployment/defectdojo -n defectdojo
kubectl rollout restart deployment/defectdojo-celery-worker -n defectdojo
kubectl rollout restart deployment/defectdojo-celery-beat -n defectdojo
```

---

## Teardown

```bash
kubectl delete namespace defectdojo
```

> This deletes all resources in the namespace including the media PVC. Back up uploaded files beforehand if needed.
