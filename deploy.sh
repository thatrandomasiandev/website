#!/bin/bash

# CometAI Website Deployment Script
# This script helps deploy your website to various cloud platforms

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="cometai-website"
IMAGE_NAME="cometai-website"
CONTAINER_NAME="cometai-website"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Docker is available"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t $IMAGE_NAME .
    log_success "Docker image built successfully"
}

# Run locally with Docker
run_local() {
    log_info "Starting local deployment..."
    
    # Stop existing container if running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log_info "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Run new container
    docker run -d \
        --name $CONTAINER_NAME \
        -p 8080:8080 \
        -v "$(pwd)/logs:/app/logs" \
        --restart unless-stopped \
        $IMAGE_NAME
    
    log_success "Container started successfully!"
    log_info "Website available at: http://localhost:8080"
    log_info "AI Chat available at: http://localhost:8080/chat-interface.html"
    log_info "API available at: http://localhost:8080/api"
}

# Deploy to DigitalOcean
deploy_digitalocean() {
    log_info "Preparing DigitalOcean deployment..."
    
    # Check if doctl is installed
    if ! command -v doctl &> /dev/null; then
        log_warning "doctl is not installed. Please install it for DigitalOcean deployment."
        log_info "Visit: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        return 1
    fi
    
    log_info "Creating docker-compose.prod.yml for DigitalOcean..."
    
    cat > docker-compose.prod.yml << EOF
version: '3.8'

services:
  cometai-website:
    image: $IMAGE_NAME:latest
    ports:
      - "80:8080"
      - "443:8080"
    environment:
      - PYTHONUNBUFFERED=1
      - MODEL_NAME=qwen2.5-coder-7b-instruct
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
EOF
    
    log_success "DigitalOcean configuration created"
    log_info "Next steps:"
    echo "1. Create a DigitalOcean Droplet (minimum 16GB RAM recommended)"
    echo "2. Upload your code to the droplet"
    echo "3. Run: docker-compose -f docker-compose.prod.yml up -d"
}

# Deploy to AWS
deploy_aws() {
    log_info "Preparing AWS deployment..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_warning "AWS CLI is not installed. Please install it for AWS deployment."
        log_info "Visit: https://aws.amazon.com/cli/"
        return 1
    fi
    
    log_info "Creating AWS deployment files..."
    
    # Create Dockerfile.aws (optimized for AWS)
    cat > Dockerfile.aws << EOF
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \\
    git curl build-essential \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \\
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["python3", "unified_web_server.py", "--host", "0.0.0.0", "--port", "8080"]
EOF
    
    # Create task definition for ECS
    cat > task-definition.json << EOF
{
    "family": "cometai-website",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "4096",
    "memory": "16384",
    "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "cometai-website",
            "image": "YOUR_ECR_REPO/cometai-website:latest",
            "portMappings": [
                {
                    "containerPort": 8080,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/cometai-website",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8080/api/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 120
            }
        }
    ]
}
EOF
    
    log_success "AWS configuration created"
    log_info "Next steps:"
    echo "1. Create an ECR repository"
    echo "2. Build and push image to ECR"
    echo "3. Update task-definition.json with your account details"
    echo "4. Create ECS cluster and service"
}

# Show logs
show_logs() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log_info "Showing container logs..."
        docker logs -f $CONTAINER_NAME
    else
        log_error "Container is not running"
    fi
}

# Stop deployment
stop_deployment() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log_info "Stopping container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        log_success "Container stopped"
    else
        log_warning "Container is not running"
    fi
}

# Show status
show_status() {
    log_info "Deployment Status:"
    
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        echo "✅ Container is running"
        echo "🌐 Website: http://localhost:8080"
        echo "🤖 Chat: http://localhost:8080/chat-interface.html"
        echo "📡 API: http://localhost:8080/api"
        
        # Test health endpoint
        if curl -s http://localhost:8080/api/health > /dev/null; then
            echo "✅ Health check: PASSED"
        else
            echo "❌ Health check: FAILED"
        fi
    else
        echo "❌ Container is not running"
    fi
}

# Main script
case "$1" in
    "build")
        check_docker
        build_image
        ;;
    "local")
        check_docker
        build_image
        run_local
        ;;
    "digitalocean")
        deploy_digitalocean
        ;;
    "aws")
        deploy_aws
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        stop_deployment
        ;;
    "status")
        show_status
        ;;
    "restart")
        stop_deployment
        sleep 2
        run_local
        ;;
    *)
        echo "CometAI Website Deployment Script"
        echo ""
        echo "Usage: $0 {build|local|digitalocean|aws|logs|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  build        - Build Docker image"
        echo "  local        - Deploy locally with Docker"
        echo "  digitalocean - Prepare DigitalOcean deployment"
        echo "  aws          - Prepare AWS deployment"
        echo "  logs         - Show container logs"
        echo "  stop         - Stop local deployment"
        echo "  status       - Show deployment status"
        echo "  restart      - Restart local deployment"
        echo ""
        echo "Examples:"
        echo "  $0 local     # Deploy locally"
        echo "  $0 status    # Check status"
        echo "  $0 logs      # View logs"
        exit 1
        ;;
esac
