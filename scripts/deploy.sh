#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
INSTANCE_DNS=""      # <<< REPLACE WITH YOUR ACTUAL EC2 INSTANCE PUBLIC IP ADDRESS
SSH_USER="ec2-user"                      # SSH username for the EC2 instance
SSH_KEY_PATH="upc-chatbot-key"              # Path to your private SSH key relative to this script
ECR_REPO_URI="538125082622.dkr.ecr.us-east-1.amazonaws.com/upc-chatbot"
CONTAINER_NAME="upc-chatbot"
IMAGE_TAG="latest"
AWS_REGION="us-east-1"
# --- End Configuration ---

# Define the image name locally
IMAGE_NAME="${ECR_REPO_URI}:${IMAGE_TAG}"
echo "Target instance: $INSTANCE_DNS"
echo "Image to deploy: [$IMAGE_NAME]" # Debug local image name

# 2. Define the commands to run remotely via SSH
# Use single quotes around EOF to prevent local expansion of variables inside the block.
# Variables will be exported explicitly before execution.
REMOTE_SCRIPT=$(cat <<'EOF'
set -ex # Exit on error within the remote execution

# These variables are expected to be exported into the environment before this script runs
echo "--- Remote Execution Start ---"
echo "Remote Environment Variables:"
echo "  REMOTE_IMAGE_NAME=[$REMOTE_IMAGE_NAME]"
  echo "  REMOTE_ECR_REGISTRY_ID=[$REMOTE_ECR_REGISTRY_ID]"
  echo "  REMOTE_AWS_REGION=[$REMOTE_AWS_REGION]"
echo "----------------------------"

# Authenticate Docker to ECR
echo "Authenticating Docker with ECR..."
aws ecr get-login-password --region "$REMOTE_AWS_REGION" | sudo docker login --username AWS --password-stdin "$REMOTE_ECR_REGISTRY_ID".dkr.ecr."$REMOTE_AWS_REGION".amazonaws.com

# Pull the latest image
echo "Pulling latest image: [$REMOTE_IMAGE_NAME]..."
sudo docker pull "$REMOTE_IMAGE_NAME"

# Find containers to clean up - collect all IDs in a single variable with de-duplication
echo "Finding containers to clean up..."
CONTAINERS_TO_CLEAN=""

# Find all relevant containers with different filters
IMAGE_CONTAINERS=$(sudo docker ps -a -q --filter ancestor="$REMOTE_IMAGE_NAME")
NAMED_CONTAINERS=$(sudo docker ps -a -q --filter name="$CONTAINER_NAME")
PORT_CONTAINERS=$(sudo docker ps -q --filter publish=8080)

# Combine all container IDs
ALL_CONTAINERS="$IMAGE_CONTAINERS $NAMED_CONTAINERS $PORT_CONTAINERS"

# De-duplicate container IDs
for ID in $ALL_CONTAINERS; do
  if [[ ! "$CONTAINERS_TO_CLEAN" =~ "$ID" ]]; then
    CONTAINERS_TO_CLEAN="$CONTAINERS_TO_CLEAN $ID"
  fi
done

# Trim leading space
CONTAINERS_TO_CLEAN=$(echo $CONTAINERS_TO_CLEAN | xargs)

# If we found containers, stop and remove them
if [ ! -z "$CONTAINERS_TO_CLEAN" ]; then
  echo "Found containers to clean up: [$CONTAINERS_TO_CLEAN]"
  
  # Stop all containers
  echo "Stopping containers..."
  for ID in $CONTAINERS_TO_CLEAN; do
    sudo docker stop $ID || echo "Warning: Could not stop container $ID"
  done
  
  # Remove all containers
  echo "Removing containers..."
  for ID in $CONTAINERS_TO_CLEAN; do
    sudo docker rm $ID || echo "Warning: Could not remove container $ID"
  done
  
  # Additional safety: wait a moment for port to be released
  sleep 3
else
  echo "No existing containers found to clean up."
fi

# One last safety check to make sure port 8080 is available
if sudo netstat -tulpn | grep ":8080"; then
  echo "WARNING: Port 8080 is still in use by some process!"
  echo "Attempting to kill any process using port 8080..."
  PORT_PID=$(sudo lsof -t -i:8080)
  if [ ! -z "$PORT_PID" ]; then
    echo "Killing process with PID $PORT_PID"
    sudo kill -9 $PORT_PID
    sleep 2
  fi
fi

# Run the new container
echo "Starting new container from [$REMOTE_IMAGE_NAME]..."
# Use double quotes around the image name argument
sudo docker run -d -p 8080:8080 --restart always --name "$CONTAINER_NAME" "$REMOTE_IMAGE_NAME"

echo "--- Remote Execution Finished ---"
EOF
)

# 3. Execute the commands via SSH
echo "Connecting via SSH to $SSH_USER@$INSTANCE_DNS and executing deployment commands..."
# Ensure the key has correct permissions (optional but good practice)
chmod 600 "$SSH_KEY_PATH"

# Add debug line
echo "DEBUG: SSH Host string is: [$SSH_USER@$INSTANCE_DNS]"
echo "DEBUG: Preparing to pipe script. Local IMAGE_NAME=[$IMAGE_NAME]" # Debug local name again

# Extract registry ID locally
ECR_REGISTRY_ID=$(echo "$ECR_REPO_URI" | cut -d. -f1)

# Execute the commands by piping them to bash on the remote host
# Export variables needed by the remote script first, then pipe the script itself.
{
    echo "export REMOTE_IMAGE_NAME='$IMAGE_NAME'" # Use single quotes for literal value
    echo "export REMOTE_ECR_REGISTRY_ID='$ECR_REGISTRY_ID'"
    echo "export REMOTE_AWS_REGION='$AWS_REGION'"
    echo "$REMOTE_SCRIPT"
} | ssh -i "$SSH_KEY_PATH" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "$SSH_USER@$INSTANCE_DNS" \
    bash

echo "Deployment script finished successfully."
exit 0
