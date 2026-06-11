kubectl create namespace ollama
kubectl apply -f ollama-orchestrator-pvc.yaml
kubectl apply -f ollama-orchestrator-deployment.yaml
kubectl apply -f ollama-orchestrator-svc.yaml