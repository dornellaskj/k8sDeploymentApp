#!/bin/bash
# Run this on each MicroK8s node to configure it to pull from Harbor over HTTPS.
# Usage: sudo bash setup-node.sh <TRAEFIK_METALLB_IP> <PATH_TO_HARBOR_CERT>

set -e

HARBOR_HOST="harbor.kevin.local"
TRAEFIK_IP="${1:?Usage: $0 <TRAEFIK_METALLB_IP> <PATH_TO_HARBOR_CERT>}"
CERT_FILE="${2:?Usage: $0 <TRAEFIK_METALLB_IP> <PATH_TO_HARBOR_CERT>}"

# Validate certificate file exists
if [ ! -f "$CERT_FILE" ]; then
  echo "Error: Certificate file $CERT_FILE not found"
  exit 1
fi

# Add DNS entry if not already present
if ! grep -q "$HARBOR_HOST" /etc/hosts; then
  echo "$TRAEFIK_IP $HARBOR_HOST" >> /etc/hosts
  echo "Added $HARBOR_HOST -> $TRAEFIK_IP to /etc/hosts"
fi

# Configure containerd to trust Harbor's certificate
CERTS_DIR="/var/snap/microk8s/current/args/certs.d/$HARBOR_HOST"
mkdir -p "$CERTS_DIR"

# Copy certificate to containerd's trust store
cp "$CERT_FILE" "$CERTS_DIR/ca.crt"
echo "Copied certificate to $CERTS_DIR/ca.crt"

cat > "$CERTS_DIR/hosts.toml" <<EOF
server = "https://$HARBOR_HOST"

[host."https://$HARBOR_HOST"]
  capabilities = ["pull", "resolve", "push"]
  ca = "$CERTS_DIR/ca.crt"
EOF

echo "Wrote $CERTS_DIR/hosts.toml"

# Restart MicroK8s to pick up containerd changes
echo "Restarting MicroK8s..."
microk8s stop
microk8s start

echo "Done. Verify with: microk8s ctr image pull $HARBOR_HOST/library/test:latest"
