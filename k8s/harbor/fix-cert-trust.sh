#!/bin/bash
# Fix MicroK8s containerd certificate trust for Harbor

set -e

HARBOR_HOST="harbor.kevin.local"
CERT_DIR="/var/snap/microk8s/current/args/certs.d/${HARBOR_HOST}"
CERT_FILE="$HOME/repos/k8sDeploymentApp/k8s/harbor/certs/tls.crt"

echo "=== Configuring MicroK8s to trust Harbor certificate ==="

# Check if certificate exists
if [ ! -f "$CERT_FILE" ]; then
    echo "ERROR: Certificate file not found at $CERT_FILE"
    exit 1
fi

echo "✓ Certificate file found"

# Create directory for Harbor certs
echo "Creating certificate directory..."
sudo mkdir -p "${CERT_DIR}"

# Copy certificate
echo "Copying certificate..."
sudo cp "$CERT_FILE" "${CERT_DIR}/ca.crt"

# Create hosts.toml configuration
echo "Creating hosts.toml configuration..."
sudo tee "${CERT_DIR}/hosts.toml" > /dev/null <<EOF
server = "https://${HARBOR_HOST}"

[host."https://${HARBOR_HOST}"]
  capabilities = ["pull", "resolve", "push"]
  ca = "${CERT_DIR}/ca.crt"
  skip_verify = false
EOF

echo "✓ Configuration files created"

# Verify files
echo ""
echo "=== Verification ==="
ls -la "${CERT_DIR}/"
echo ""
cat "${CERT_DIR}/hosts.toml"
echo ""

# Restart MicroK8s
echo "=== Restarting MicroK8s ==="
sudo microk8s stop
sleep 3
sudo microk8s start

echo "Waiting for MicroK8s to be ready..."
microk8s status --wait-ready

echo ""
echo "=== Testing connection ==="
curl -k -u admin:Harbor12345 https://${HARBOR_HOST}/v2/_catalog

echo ""
echo "✓ Certificate trust configured successfully!"
echo ""
echo "Now you can push/pull images:"
echo "  microk8s ctr images ls | grep elasticsearch"
echo "  microk8s ctr images tag elasticsearch-webui:latest ${HARBOR_HOST}/library/elasticsearch-webui:latest"
echo ""
