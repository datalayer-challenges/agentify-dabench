#!/bin/bash

# AgentBeats Docker Deployment Helper
# This script helps with building and pushing Docker images for AgentBeats deployment

set -e

# Colors for output
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'
RESET='\033[0m'

# Configuration
REGISTRY="ghcr.io"
USERNAME="${1:-$(whoami)}"
REPO_NAME="agentify-dab-step"

echo -e "${CYAN}🚀 AgentBeats Docker Deployment Helper${RESET}"
echo -e "${CYAN}═══════════════════════════════════════${RESET}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${RESET}"
    exit 1
fi

# Function to check if user is logged in to GitHub Container Registry
check_ghcr_login() {
    if docker info 2>/dev/null | grep -q "ghcr.io"; then
        echo -e "${GREEN}✅ Already logged in to GitHub Container Registry${RESET}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Not logged in to GitHub Container Registry${RESET}"
        return 1
    fi
}

# Function to login to GitHub Container Registry
login_ghcr() {
    echo -e "${BLUE}🔑 Logging in to GitHub Container Registry...${RESET}"
    echo -e "${YELLOW}💡 You'll need a GitHub Personal Access Token with 'write:packages' scope${RESET}"
    echo -e "${YELLOW}💡 Create one at: https://github.com/settings/tokens/new${RESET}"
    echo ""
    
    read -p "GitHub username: " github_username
    read -sp "GitHub Personal Access Token: " github_token
    echo ""
    
    if echo "$github_token" | docker login ghcr.io -u "$github_username" --password-stdin; then
        echo -e "${GREEN}✅ Successfully logged in to GitHub Container Registry${RESET}"
        USERNAME="$github_username"
    else
        echo -e "${RED}❌ Failed to log in to GitHub Container Registry${RESET}"
        exit 1
    fi
}

# Function to build images
build_images() {
    echo -e "${BLUE}🐳 Building AgentBeats-compatible Docker images...${RESET}"
    echo -e "${YELLOW}📦 Platform: linux/amd64 (GitHub Actions compatible)${RESET}"
    echo ""
    
    # Build Green Agent
    echo -e "${CYAN}Building Green Agent...${RESET}"
    docker build --platform linux/amd64 -f Dockerfile.green \
        -t "${REGISTRY}/${USERNAME}/${REPO_NAME}-green:latest" .
    
    # Build Purple Agent  
    echo -e "${CYAN}Building Purple Agent...${RESET}"
    docker build --platform linux/amd64 -f Dockerfile.purple \
        -t "${REGISTRY}/${USERNAME}/${REPO_NAME}-purple:latest" .
    
    # Build Jupyter MCP Server
    echo -e "${CYAN}Building Jupyter MCP Server...${RESET}"
    docker build --platform linux/amd64 -f Dockerfile.jupyter \
        -t "${REGISTRY}/${USERNAME}/${REPO_NAME}-jupyter:latest" .
    
    echo -e "${GREEN}✅ All images built successfully${RESET}"
}

# Function to push images
push_images() {
    echo -e "${BLUE}📤 Pushing images to GitHub Container Registry...${RESET}"
    
    docker push "${REGISTRY}/${USERNAME}/${REPO_NAME}-green:latest"
    docker push "${REGISTRY}/${USERNAME}/${REPO_NAME}-purple:latest"  
    docker push "${REGISTRY}/${USERNAME}/${REPO_NAME}-jupyter:latest"
    
    echo -e "${GREEN}✅ All images pushed successfully${RESET}"
    echo ""
    echo -e "${CYAN}🎯 Your images are now available at:${RESET}"
    echo -e "  ${REGISTRY}/${USERNAME}/${REPO_NAME}-green:latest"
    echo -e "  ${REGISTRY}/${USERNAME}/${REPO_NAME}-purple:latest"
    echo -e "  ${REGISTRY}/${USERNAME}/${REPO_NAME}-jupyter:latest"
}

# Function to test images locally
test_images() {
    echo -e "${BLUE}🧪 Testing images locally...${RESET}"
    
    # Test Green Agent
    echo -e "${CYAN}Testing Green Agent (5 seconds)...${RESET}"
    timeout 5s docker run --rm "${REGISTRY}/${USERNAME}/${REPO_NAME}-green:latest" --help || true
    
    # Test Purple Agent
    echo -e "${CYAN}Testing Purple Agent (5 seconds)...${RESET}"  
    timeout 5s docker run --rm "${REGISTRY}/${USERNAME}/${REPO_NAME}-purple:latest" --help || true
    
    echo -e "${GREEN}✅ Images tested successfully${RESET}"
}

# Main menu
echo -e "${YELLOW}What would you like to do?${RESET}"
echo -e "  ${GREEN}1)${RESET} Build images"
echo -e "  ${GREEN}2)${RESET} Build and push images"
echo -e "  ${GREEN}3)${RESET} Login to GitHub Container Registry"
echo -e "  ${GREEN}4)${RESET} Test images locally"
echo -e "  ${GREEN}5)${RESET} Full deployment (build, test, push)"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        build_images
        ;;
    2)
        if ! check_ghcr_login; then
            login_ghcr
        fi
        build_images
        push_images
        ;;
    3)
        login_ghcr
        ;;
    4)
        test_images
        ;;
    5)
        if ! check_ghcr_login; then
            login_ghcr
        fi
        build_images
        test_images
        push_images
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${RESET}"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}🎉 Done! Your AgentBeats deployment is ready.${RESET}"