#!/bin/bash
set -e

# Set your domain here
DOMAIN="upcchatbot.maxellmilay.com"

# Log function for better debugging
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/cloud-init-custom.log
}

log "Starting initialization script"

# Update packages and install necessary tools
log "Updating packages"
dnf update -y || log "Package update failed, but continuing"

log "Installing required packages"
dnf install -y nginx || log "Some package installation failed, but continuing"

# Install Docker separately to handle potential failures
log "Installing Docker"
dnf install -y docker || log "Docker installation failed, but continuing"

# Install AWS CLI separately
log "Installing AWS CLI"
dnf install -y aws-cli || log "AWS CLI installation failed, but continuing"

# Install Certbot (for Let's Encrypt)
log "Installing Certbot"
dnf install -y certbot python3-certbot-nginx || log "Certbot installation failed, but continuing"

# Create nginx config directories if they don't exist
log "Creating Nginx config directories"
mkdir -p /etc/nginx/conf.d/

# Start and enable Nginx
log "Starting Nginx"
systemctl enable nginx
systemctl start nginx || log "Failed to start Nginx, but continuing"

# Start Docker service
log "Starting Docker service"
systemctl enable docker
systemctl start docker || log "Failed to start Docker, but continuing"

# Authenticate to ECR with retry logic
log "Authenticating to ECR"
max_attempts=3
attempt=0
while [ $attempt -lt $max_attempts ]; do
  attempt=$((attempt+1))
  log "ECR login attempt $attempt of $max_attempts"
  if aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 538125082622.dkr.ecr.us-east-1.amazonaws.com; then
    log "ECR authentication successful"
    break
  else
    log "ECR authentication failed"
    if [ $attempt -eq $max_attempts ]; then
      log "All ECR authentication attempts failed, but continuing"
    else
      sleep 5
    fi
  fi
done

# Pull the Docker image from ECR with retry logic
log "Pulling Docker image"
max_attempts=3
attempt=0
while [ $attempt -lt $max_attempts ]; do
  attempt=$((attempt+1))
  log "Docker pull attempt $attempt of $max_attempts"
  if docker pull 538125082622.dkr.ecr.us-east-1.amazonaws.com/upc-chatbot-backend; then
    log "Docker image pull successful"
    break
  else
    log "Docker image pull failed"
    if [ $attempt -eq $max_attempts ]; then
      log "All Docker pull attempts failed, but continuing"
    else
      sleep 5
    fi
  fi
done

# Run the Docker image, exposing the app on 127.0.0.1:8080
log "Running Docker container"
docker run -d -p 8080:8080 --restart always 538125082622.dkr.ecr.us-east-1.amazonaws.com/upc-chatbot-backend || log "Failed to run Docker container, but continuing"

# Wait for container to be fully running
log "Waiting for Docker container to start"
sleep 5

# Check if Docker container is running
log "Checking Docker container status"
if docker ps | grep upc-chatbot-backend; then
  log "Docker container is running"
else
  log "Docker container is not running, attempting to start again"
  docker run -d -p 8080:8080 --restart always 538125082622.dkr.ecr.us-east-1.amazonaws.com/upc-chatbot-backend
fi

# Test if the local Docker container is accessible
log "Testing local Docker container"
max_attempts=5
attempt=0
while [ $attempt -lt $max_attempts ]; do
  attempt=$((attempt+1))
  log "Testing Docker container accessibility, attempt $attempt of $max_attempts"
  if curl -s http://127.0.0.1:8080 > /dev/null; then
    log "Local Docker container is accessible"
    break
  else
    log "Cannot access local Docker container on port 8080, waiting..."
    if [ $attempt -eq $max_attempts ]; then
      log "All attempts to connect to Docker container failed, but continuing"
    else
      sleep 10
    fi
  fi
done

# Configure Nginx as a reverse proxy
log "Configuring Nginx"
cat <<EOF > /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Restart Nginx to apply the configuration
log "Restarting Nginx with new configuration"
systemctl restart nginx

# Wait for Nginx to fully start
sleep 5

# Check if Nginx is running
if systemctl is-active --quiet nginx; then
  log "Nginx is running"
else
  log "Nginx is not running, attempting to start"
  systemctl start nginx
fi

# Get instance IP and check DNS resolution
log "Checking DNS resolution"
INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
DOMAIN_IP=$(dig +short ${DOMAIN} || echo "")
log "Instance IP: $INSTANCE_IP, Domain resolves to: $DOMAIN_IP"

# Ensure security group allows HTTP and HTTPS
log "Checking for HTTP port accessibility"
if curl -s localhost:80 > /dev/null; then
  log "HTTP (port 80) is accessible"
else
  log "HTTP (port 80) may not be accessible - check your security group settings"
  log "Waiting 30 seconds before trying to obtain SSL certificate..."
  sleep 30
fi

# Issue SSL cert from Let's Encrypt 
log "Setting up SSL with certbot"
max_attempts=3
attempt=0
while [ $attempt -lt $max_attempts ]; do
  attempt=$((attempt+1))
  log "Certbot SSL attempt $attempt of $max_attempts"
  
  if [ "$DOMAIN_IP" = "$INSTANCE_IP" ] || [ "$attempt" -gt 1 ]; then
    if certbot --nginx --non-interactive --agree-tos --redirect \
        --email maxell.milay@gmail.com \
        -d ${DOMAIN}; then
      log "Successfully obtained SSL certificate"
      break
    else
      log "Failed to obtain SSL certificate on attempt $attempt"
      if [ $attempt -eq $max_attempts ]; then
        log "All SSL certificate attempts failed. You may need to manually run:"
        log "certbot --nginx --non-interactive --agree-tos --redirect --email maxell.milay@gmail.com -d ${DOMAIN}"
      else
        log "Waiting 60 seconds before retry..."
        sleep 60
      fi
    fi
  else
    log "WARNING: Domain $DOMAIN does not point to this instance IP ($INSTANCE_IP)"
    log "Waiting 60 seconds to see if DNS propagates before retry..."
    sleep 60
    # Update domain IP for next check
    DOMAIN_IP=$(dig +short ${DOMAIN} || echo "")
    log "Updated Domain IP: $DOMAIN_IP"
  fi
done

# Final status check
log "Performing final status check"
systemctl status nginx | grep Active

log "Setup complete - now test your domain: https://${DOMAIN}"
log "If you're still seeing issues, please check:"
log "1. Your DNS records - make sure ${DOMAIN} points to ${INSTANCE_IP}"
log "2. Your EC2 security group - make sure ports 80 and 443 are open"
log "3. Check Nginx error logs: journalctl -u nginx"
