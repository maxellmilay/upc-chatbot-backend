"""
Example usage of the S3 service for downloading files and other S3 operations.

Before using this service, make sure to set the following environment variables:
- AWS_ACCESS_KEY_ID: Your AWS access key ID
- AWS_SECRET_ACCESS_KEY: Your AWS secret access key
- AWS_REGION: Your AWS region (optional, defaults to 'us-east-1')
- AWS_S3_BUCKET: Your default S3 bucket name (optional)

Or add them to your Django settings.py file:
AWS_ACCESS_KEY_ID = 'your_access_key_id'
AWS_SECRET_ACCESS_KEY = 'your_secret_access_key'
AWS_REGION = 'us-east-1'
AWS_S3_BUCKET = 'your-bucket-name'
"""

from main.services.s3 import S3Service
import os
from dotenv import load_dotenv


def example_download_operations():
    """
    Example demonstrating various S3 download operations.
    """
    # Get S3 service instance
    s3_service = S3Service()
    
    # Example: Download file to memory
    print("\nExample: Downloading file to memory")
    content = s3_service.download_file_to_memory(
        key='documents/sample.pdf',
        bucket_name='my-bucket'
    )
    if content:
        print(f"✅ File downloaded to memory! Size: {len(content)} bytes")
        # You can now process the content directly
        # For example, save it with a different name:
        # with open('/tmp/sample_from_memory.pdf', 'wb') as f:
        #     f.write(content)
    else:
        print("❌ Failed to download file to memory")


if __name__ == "__main__":
    load_dotenv()
    print("🚀 S3 Service Usage Example\n")
    print("=" * 50)
    
    # Check if AWS credentials are configured
    if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')):
        print("⚠️  AWS credentials not found in environment variables.")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY before running these examples.")
        print("\nYou can set them like this:")
        print("export AWS_ACCESS_KEY_ID='your_access_key_id'")
        print("export AWS_SECRET_ACCESS_KEY='your_secret_access_key'")
        print("export AWS_S3_BUCKET='your_bucket_name'  # Optional")
        exit(1)
    
    try:
        example_download_operations()
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"❌ An error occurred while running examples: {e}")
        print("Make sure your AWS credentials are correct and you have access to the specified S3 bucket.")
