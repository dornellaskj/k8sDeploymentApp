#!/bin/bash
# Generate self-signed TLS certificate for Harbor
# This creates a certificate valid for 365 days for harbor.kevin.local

set -e

DOMAIN="harbor.kevin.local"
NAMESPACE="harbor"
SECRET_NAME="harbor-tls-secret"
CERT_DIR="./certs"

echo "Generating self-signed TLS certificate for $DOMAIN..."

# Create certs directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Generate private key
openssl genrsa -out "$CERT_DIR/tls.key" 2048

# Generate certificate signing request
openssl req -new -key "$CERT_DIR/tls.key" -out "$CERT_DIR/tls.csr" \
  -subj "/CN=$DOMAIN/O=Harbor/C=US"

# Generate self-signed certificate valid for 365 days
openssl x509 -req -days 365 -in "$CERT_DIR/tls.csr" \
  -signkey "$CERT_DIR/tls.key" -out "$CERT_DIR/tls.crt" \
  -extfile <(printf "subjectAltName=DNS:$DOMAIN,DNS:*.$DOMAIN")

echo "Certificate generated successfully!"
echo "Certificate: $CERT_DIR/tls.crt"
echo "Private Key: $CERT_DIR/tls.key"

# Create Kubernetes secret
echo ""
echo "Creating Kubernetes secret $SECRET_NAME in namespace $NAMESPACE..."

# Create namespace if it doesn't exist
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Delete existing secret if present
kubectl delete secret "$SECRET_NAME" -n "$NAMESPACE" --ignore-not-found=true

# Create new secret
kubectl create secret tls "$SECRET_NAME" \
  --cert="$CERT_DIR/tls.crt" \
  --key="$CERT_DIR/tls.key" \
  -n "$NAMESPACE"

echo "Secret $SECRET_NAME created successfully in namespace $NAMESPACE"
echo ""
echo "Next steps:"
echo "1. Update Harbor Helm values to enable TLS"
echo "2. Update Harbor IngressRoute to use websecure entrypoint"
echo "3. Upgrade Harbor deployment"
echo "4. Configure Docker client to trust the certificate"
echo ""
echo "To export certificate for Docker:"
echo "  cp $CERT_DIR/tls.crt /etc/docker/certs.d/$DOMAIN/ca.crt"
