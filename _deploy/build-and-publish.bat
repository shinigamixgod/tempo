@echo off
setlocal enabledelayedexpansion

:: Replace with your Docker Hub username
set DOCKERHUB_USER=geoglify

:: Check if the first argument is "build"
if /I "%1"=="build" (
    echo 🔨 Building images using docker-compose.build.yml...
    docker compose -f docker-compose.build.yml build
) else (
    echo ⚠️ Skipping build step. To build, run: build-and-publish.bat build
)

:: List of custom image services
set IMAGES=tempo

for %%I in (%IMAGES%) do (
    echo 🚀 Pushing !DOCKERHUB_USER!/%%I:latest to Docker Hub...
    docker push !DOCKERHUB_USER!/%%I:latest
)

echo ✅ Done.
pause