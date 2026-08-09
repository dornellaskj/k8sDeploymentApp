#!/bin/bash
# Run this on each MicroK8s node to configure it to pull from Harbor over HTTP.
# Usage: sudo bash setup-node.sh <TRAEFIK_METALLB_IP>

set -e

HARBOR_HOST="harbor.kevin.local"
TRAEFIK_IP="${1:?Usage: $0 <TRAEFIK_METALLB_IP>}"

# Add DNS entry if not already present
if ! grep -q "$HARBOR_HOST" /etc/hosts; then
  echo "$TRAEFIK_IP $HARBOR_HOST" >> /etc/hosts
  echo "Added $HARBOR_HOST -> $TRAEFIK_IP to /etc/hosts"
fi

# Configure containerd to allow insecure HTTP pulls from Harbor
CERTS_DIR="/var/snap/microk8s/current/args/certs.d/$HARBOR_HOST"
mkdir -p "$CERTS_DIR"

cat > "$CERTS_DIR/hosts.toml" <<EOF
server = "http://$HARBOR_HOST"

[host."http://$HARBOR_HOST"]
  capabilities = ["pull", "resolve", "push"]
  skip_verify = true
EOF

echo "Wrote $CERTS_DIR/hosts.toml"

# Restart MicroK8s to pick up containerd changes
echo "Restarting MicroK8s..."
microk8s stop
microk8s start

echo "Done. Verify with: microk8s ctr image pull $HARBOR_HOST/globoticket/test:latest"
