#!/bin/bash
set -e

REPO="onesvat/whisper-server"
VERSION="${1:-latest}"

echo "Building Docker images..."

# GPU version
docker build -f Dockerfile.gpu -t "${REPO}:gpu" -t "${REPO}:gpu-${VERSION}" .

# CPU version
docker build -f Dockerfile.cpu -t "${REPO}:cpu" -t "${REPO}:cpu-${VERSION}" .

# Latest tag (GPU)
docker tag "${REPO}:gpu" "${REPO}:latest"

echo "Images built successfully:"
docker images "${REPO}"

echo ""
echo "Pushing to Docker Hub..."

# Push all tags
docker push "${REPO}:gpu"
docker push "${REPO}:gpu-${VERSION}"
docker push "${REPO}:cpu"
docker push "${REPO}:cpu-${VERSION}"
docker push "${REPO}:latest"

echo ""
echo "✅ All images pushed successfully!"
echo "Available images:"
echo "  - ${REPO}:gpu"
echo "  - ${REPO}:gpu-${VERSION}"
echo "  - ${REPO}:cpu"
echo "  - ${REPO}:cpu-${VERSION}"
echo "  - ${REPO}:latest"
