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

from services.s3 import get_s3_service
import os


def example_download_operations():
    """
    Example demonstrating various S3 download operations.
    """
    # Get S3 service instance
    s3_service = get_s3_service()
    
    # Example 1: Download file to local filesystem
    print("Example 1: Downloading file to local filesystem")
    success = s3_service.download_file(
        key='documents/sample.pdf',
        local_path='/tmp/downloaded_sample.pdf',
        bucket_name='my-bucket'  # Optional if you have a default bucket configured
    )
    if success:
        print("✅ File downloaded successfully!")
    else:
        print("❌ Failed to download file")
    
    # Example 2: Download file to memory
    print("\nExample 2: Downloading file to memory")
    content = s3_service.download_file_to_memory(
        key='documents/sample.pdf',
        bucket_name='my-bucket'
    )
    if content:
        print(f"✅ File downloaded to memory! Size: {len(content)} bytes")
        # You can now process the content directly
        # For example, save it with a different name:
        with open('/tmp/sample_from_memory.pdf', 'wb') as f:
            f.write(content)
    else:
        print("❌ Failed to download file to memory")
    
    # Example 3: Check if file exists before downloading
    print("\nExample 3: Checking if file exists")
    file_key = 'documents/sample.pdf'
    if s3_service.file_exists(file_key, 'my-bucket'):
        print(f"✅ File {file_key} exists in S3")
        
        # Get file metadata
        metadata = s3_service.get_file_metadata(file_key, 'my-bucket')
        if metadata:
            print(f"📄 File size: {metadata['size']} bytes")
            print(f"📅 Last modified: {metadata['last_modified']}")
            print(f"🔖 Content type: {metadata['content_type']}")
    else:
        print(f"❌ File {file_key} does not exist in S3")
    
    # Example 4: Generate presigned download URL
    print("\nExample 4: Generating presigned download URL")
    download_url = s3_service.generate_presigned_download_url(
        key='documents/sample.pdf',
        expiration=3600,  # 1 hour
        bucket_name='my-bucket'
    )
    if download_url:
        print(f"✅ Presigned URL generated: {download_url}")
        print("🔗 You can use this URL to download the file directly from a browser or HTTP client")
    else:
        print("❌ Failed to generate presigned URL")


def example_with_error_handling():
    """
    Example demonstrating proper error handling with the S3 service.
    """
    s3_service = get_s3_service()
    
    try:
        # Attempt to download a file that might not exist
        success = s3_service.download_file(
            key='nonexistent/file.txt',
            local_path='/tmp/nonexistent_file.txt'
        )
        
        if not success:
            print("⚠️ Download failed - this is expected for demonstration")
            
            # Check what went wrong by trying to get metadata
            metadata = s3_service.get_file_metadata('nonexistent/file.txt')
            if metadata is None:
                print("📝 File does not exist or is inaccessible")
    
    except Exception as e:
        print(f"❌ An error occurred: {e}")


def example_bulk_download():
    """
    Example of downloading multiple files.
    """
    s3_service = get_s3_service()
    
    files_to_download = [
        {'key': 'documents/file1.pdf', 'local_path': '/tmp/file1.pdf'},
        {'key': 'documents/file2.pdf', 'local_path': '/tmp/file2.pdf'},
        {'key': 'images/photo1.jpg', 'local_path': '/tmp/photo1.jpg'},
    ]
    
    print("Example: Bulk download")
    successful_downloads = 0
    
    for file_info in files_to_download:
        print(f"📥 Downloading {file_info['key']}...")
        success = s3_service.download_file(
            key=file_info['key'],
            local_path=file_info['local_path']
        )
        
        if success:
            successful_downloads += 1
            print(f"✅ Downloaded {file_info['key']}")
        else:
            print(f"❌ Failed to download {file_info['key']}")
    
    print(f"\n📊 Summary: {successful_downloads}/{len(files_to_download)} files downloaded successfully")


if __name__ == "__main__":
    print("🚀 S3 Service Usage Examples\n")
    print("=" * 50)
    
    # Check if AWS credentials are configured
    if not (os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY')):
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
        example_with_error_handling()
        print("\n" + "=" * 50)
        example_bulk_download()
        
    except Exception as e:
        print(f"❌ An error occurred while running examples: {e}")
        print("Make sure your AWS credentials are correct and you have access to the specified S3 bucket.") 