# Harbor HTTPS Setup - Deployment Instructions

This guide walks through configuring Harbor to respond to HTTPS requests and configuring Docker clients to authenticate properly.

## Prerequisites

- kubectl configured to access your Kubernetes cluster
- Harbor namespace exists in Kubernetes
- Traefik ingress controller deployed with websecure entrypoint (port 443)
- Helm installed (for Harbor deployment/upgrade)
- Access to the server running Docker client

## Step 1: Generate TLS Certificate and Kubernetes Secret

Run the certificate generation script on a machine with kubectl access:

```bash
cd k8s/harbor
chmod +x generate-tls-cert.sh
./generate-tls-cert.sh
```

This will:
- Generate a self-signed certificate for `harbor.kevin.local`
- Create a Kubernetes secret `harbor-tls-secret` in the `harbor` namespace
- Save certificate files in `./certs/` directory

**Important:** Keep the certificate files (especially `certs/tls.crt`) - you'll need them for Docker client configuration.

## Step 2: Verify Configuration Files

The following files have been updated to enable HTTPS:

1. **values.yaml** - TLS enabled, externalURL changed to HTTPS
2. **ingress.yaml** - Changed from `web` to `websecure` entrypoint, added TLS secret reference

Verify the changes:
```bash
git diff k8s/harbor/values.yaml
git diff k8s/harbor/ingress.yaml
```

## Step 3: Deploy/Upgrade Harbor

If Harbor is already deployed, upgrade it:

```bash
# Get current Harbor release name
helm list -n harbor

# Upgrade Harbor with new values
helm upgrade harbor harbor/harbor \
  -n harbor \
  -f k8s/harbor/values.yaml \
  --wait
```

If this is a fresh installation:

```bash
# Add Harbor Helm repository
helm repo add harbor https://helm.goharbor.io
helm repo update

# Install Harbor
helm install harbor harbor/harbor \
  -n harbor \
  --create-namespace \
  -f k8s/harbor/values.yaml \
  --wait
```

## Step 4: Apply IngressRoute

Apply the updated IngressRoute configuration:

```bash
kubectl apply -f k8s/harbor/ingress.yaml
```

Verify the IngressRoute:
```bash
kubectl get ingressroute harbor -n harbor -o yaml
```

## Step 5: Verify Harbor Pods

Check that all Harbor pods are running:

```bash
kubectl get pods -n harbor
```

Wait until all pods show `Running` status and `Ready 1/1`.

## Step 6: Test HTTPS Access

From your workstation/server:

```bash
# Test with curl (using -k to skip certificate verification initially)
curl -k https://harbor.kevin.local

# You should see HTML content from Harbor portal
```

## Step 7: Configure Docker Client to Trust Certificate

On each machine that needs to pull from Harbor (including Kubernetes nodes):

### Option A: Linux/Ubuntu

```bash
# Create directory for Harbor certificates
sudo mkdir -p /etc/docker/certs.d/harbor.kevin.local

# Copy the certificate (replace path with actual location)
sudo cp k8s/harbor/certs/tls.crt /etc/docker/certs.d/harbor.kevin.local/ca.crt

# Restart Docker daemon
sudo systemctl restart docker
```

### Option B: macOS

```bash
# Create directory for Harbor certificates
mkdir -p ~/.docker/certs.d/harbor.kevin.local

# Copy the certificate
cp k8s/harbor/certs/tls.crt ~/.docker/certs.d/harbor.kevin.local/ca.crt

# Restart Docker Desktop
```

### Option C: Windows

```powershell
# Create directory for Harbor certificates
New-Item -ItemType Directory -Force -Path "$env:ProgramData\docker\certs.d\harbor.kevin.local"

# Copy the certificate
Copy-Item "k8s\harbor\certs\tls.crt" "$env:ProgramData\docker\certs.d\harbor.kevin.local\ca.crt"

# Restart Docker service
Restart-Service docker
```

## Step 8: Test Docker Login

Now test Docker login:

```bash
docker login harbor.kevin.local
```

You should be prompted for username and password (default: admin / Harbor12345).

If successful, you should see:
```
Login Succeeded
```

## Step 9: Configure Kubernetes Nodes (Optional)

If you're using MicroK8s and need nodes to pull from Harbor:

```bash
# On each node, run the updated setup script
sudo bash k8s/harbor/setup-node.sh <TRAEFIK_IP> k8s/harbor/certs/tls.crt
```

Replace `<TRAEFIK_IP>` with your Traefik LoadBalancer IP (e.g., 192.168.86.148).

## Step 10: Test Image Operations

Test pushing and pulling images:

```bash
# Tag an image
docker tag hello-world:latest harbor.kevin.local/library/hello-world:latest

# Push to Harbor
docker push harbor.kevin.local/library/hello-world:latest

# Remove local image
docker rmi harbor.kevin.local/library/hello-world:latest

# Pull from Harbor
docker pull harbor.kevin.local/library/hello-world:latest
```

All operations should complete without HTTPS errors.

## Verification Checklist

- [ ] Certificate secret exists: `kubectl get secret harbor-tls-secret -n harbor`
- [ ] All Harbor pods running: `kubectl get pods -n harbor`
- [ ] IngressRoute configured: `kubectl get ingressroute harbor -n harbor`
- [ ] HTTPS accessible: `curl -k https://harbor.kevin.local` returns content
- [ ] Docker login succeeds: `docker login harbor.kevin.local`
- [ ] Image push works: `docker push harbor.kevin.local/library/test:latest`
- [ ] Image pull works: `docker pull harbor.kevin.local/library/test:latest`

## Troubleshooting

### Docker still shows "http: server gave HTTP response to HTTPS client"

**Cause:** Docker is not trusting the certificate or the certificate is in the wrong location.

**Solution:**
1. Verify certificate file exists: `ls -l /etc/docker/certs.d/harbor.kevin.local/ca.crt`
2. Restart Docker daemon: `sudo systemctl restart docker`
3. Check Docker is using the correct registry format (no port): `docker login harbor.kevin.local` (NOT `harbor.kevin.local:443`)

### Harbor pods not starting

**Cause:** TLS configuration issue or missing secret.

**Solution:**
1. Check secret exists: `kubectl get secret harbor-tls-secret -n harbor`
2. View pod logs: `kubectl logs -n harbor <pod-name>`
3. Check Harbor core pod: `kubectl describe pod -n harbor -l component=core`

### "x509: certificate signed by unknown authority"

**Cause:** Certificate not trusted by Docker client.

**Solution:**
1. Verify certificate is in correct location
2. Ensure filename is exactly `ca.crt`
3. Restart Docker daemon after copying certificate

### Cannot access harbor.kevin.local

**Cause:** DNS not resolving or routing issue.

**Solution:**
1. Add to `/etc/hosts`: `192.168.86.148 harbor.kevin.local`
2. Verify Traefik LoadBalancer IP: `kubectl get svc traefik -o wide`
3. Test connectivity: `ping harbor.kevin.local`

### Certificate expired

**Cause:** Self-signed certificate valid for 365 days has expired.

**Solution:**
1. Re-run `generate-tls-cert.sh` to create a new certificate
2. Re-deploy Harbor: `helm upgrade harbor harbor/harbor -n harbor -f k8s/harbor/values.yaml`
3. Re-configure Docker clients with new certificate

## Security Considerations

1. **Self-signed Certificate:** This setup uses a self-signed certificate, which is suitable for development/internal use. For production, consider using Let's Encrypt with cert-manager.

2. **Certificate Rotation:** Set a calendar reminder to regenerate the certificate before it expires (365 days from creation).

3. **Private Key Security:** The private key (`certs/tls.key`) should be kept secure and not committed to version control. Add `k8s/harbor/certs/` to `.gitignore`.

4. **Admin Password:** Change the default Harbor admin password after first login.

## Next Steps

1. **Create Projects:** Log into Harbor UI at `https://harbor.kevin.local` and create projects for your images
2. **Configure Image Scanning:** Enable Trivy scanning in Harbor for vulnerability detection
3. **Set up RBAC:** Create user accounts and configure access control for projects
4. **Integrate with CI/CD:** Update your CI/CD pipelines to push images to Harbor
