#!/bin/bash
set -e

echo "Building base Docker images..."
docker build -f backend/Dockerfile.base -t robotdoc-backend-base ./backend
docker build -f llama/Dockerfile.base -t robotdoc-llama-base ./llama
docker build -f llava/Dockerfile.base -t robotdoc-llava-base ./llava

echo "Starting full stack with Docker Compose..."
docker compose up --build
