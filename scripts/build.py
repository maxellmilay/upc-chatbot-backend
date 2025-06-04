#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    if Path(".env").exists():
        load_dotenv()
    else:
        print(
            "Error: .env file not found. Please create a .env file with the necessary variables."
        )
        sys.exit(1)

    # Verify required variables are set
    required_vars = [
        "AWS_ACCOUNT_ID",
        "AWS_REGION",
        "ECR_REPOSITORY",
        "IMAGE_TAG",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    for var in required_vars:
        if not os.environ.get(var):
            print(f"Error: {var} is not set in .env")
            sys.exit(1)

    # Construct the ECR repository URI
    aws_account_id = os.environ["AWS_ACCOUNT_ID"]
    aws_region = os.environ["AWS_REGION"]
    ecr_repository = os.environ["ECR_REPOSITORY"]
    image_tag = os.environ["IMAGE_TAG"]

    ecr_uri = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{ecr_repository}"

    print("Logging in to AWS ECR...")
    login_command = ["aws", "ecr", "get-login-password", "--region", aws_region]

    try:
        password = subprocess.check_output(login_command).decode("utf-8").strip()
        docker_login = subprocess.run(
            [
                "docker",
                "login",
                "--username",
                "AWS",
                "--password-stdin",
                f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com",
            ],
            input=password.encode("utf-8"),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during AWS ECR login: {e}")
        sys.exit(1)

    # Optional: Create the repository if it doesn't exist
    try:
        subprocess.run(
            [
                "aws",
                "ecr",
                "describe-repositories",
                "--repository-names",
                ecr_repository,
                "--region",
                aws_region,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"ECR repository {ecr_repository} not found. Creating repository...")
        try:
            subprocess.run(
                [
                    "aws",
                    "ecr",
                    "create-repository",
                    "--repository-name",
                    ecr_repository,
                    "--region",
                    aws_region,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error creating repository: {e}")
            sys.exit(1)

    print("Building Docker image...")
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-t",
                f"{ecr_repository}",
                ".",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e}")
        sys.exit(1)

    print("Tagging Docker image...")
    try:
        subprocess.run(
            ["docker", "tag", f"{ecr_repository}:latest", f"{ecr_uri}:{image_tag}"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error tagging Docker image: {e}")
        sys.exit(1)

    print("Pushing Docker image to ECR...")
    try:
        subprocess.run(["docker", "push", f"{ecr_uri}:{image_tag}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error pushing Docker image: {e}")
        sys.exit(1)

    print(f"Docker image pushed successfully to {ecr_uri}:{image_tag}")


if __name__ == "__main__":
    main()
