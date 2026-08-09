# Quick Start Commands - Harbor HTTPS Setup

Run these commands on your server to enable HTTPS for Harbor.

## 1. Generate Certificate and Create Secret

```bash
cd k8s/harbor
chmod +x generate-tls-cert.sh
./generate-tls-cert.sh
```

## 2. Upgrade Harbor Deployment

```bash
# If Harbor is already deployed:
helm upgrade harbor harbor/harbor -n harbor -f k8s/harbor/values.yaml --wait

# If fresh install:
# helm repo add harbor https://helm.goharbor.io
# helm install harbor harbor/harbor -n harbor --create-namespace -f k8s/harbor/values.yaml --wait
```

## 3. Apply IngressRoute

```bash
kubectl apply -f k8s/harbor/ingress.yaml
```

## 4. Verify Deployment

```bash
# Check pods are running
kubectl get pods -n harbor

# Test HTTPS access
curl -k https://harbor.kevin.local
```

## 5. Configure Docker Client

On the machine where you'll run `docker login` (replace paths as needed):

```bash
# Linux/Ubuntu
sudo mkdir -p /etc/docker/certs.d/harbor.kevin.local
sudo cp k8s/harbor/certs/tls.crt /etc/docker/certs.d/harbor.kevin.local/ca.crt
sudo systemctl restart docker
```

## 6. Test Docker Login

```bash
docker login harbor.kevin.local
# Username: admin
# Password: Harbor12345
```

## 7. Test Image Push/Pull

```bash
docker tag hello-world:latest harbor.kevin.local/library/hello-world:test
docker push harbor.kevin.local/library/hello-world:test
docker pull harbor.kevin.local/library/hello-world:test
```

---

**For detailed instructions and troubleshooting, see [HTTPS-SETUP.md](HTTPS-SETUP.md)**
