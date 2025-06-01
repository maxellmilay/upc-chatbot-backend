import os
import logging
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from django.conf import settings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class S3Service:
    """
    Service class for handling AWS S3 operations including file downloads.
    """
    
    def __init__(self):
        """
        Initialize S3 service with AWS credentials and configuration.
        """
        load_dotenv()
        
        self.aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', os.environ.get('AWS_ACCESS_KEY_ID'))
        self.aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', os.environ.get('AWS_SECRET_ACCESS_KEY'))
        self.aws_region = getattr(settings, 'AWS_REGION', os.environ.get('AWS_REGION', 'us-east-1'))
        self.aws_s3_bucket = getattr(settings, 'AWS_S3_BUCKET', os.environ.get('AWS_S3_BUCKET'))
        
        self._client = None
        self._resource = None
    
    @property
    def client(self):
        """
        Lazy initialization of S3 client.
        """
        if self._client is None:
            try:
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region
                )
            except NoCredentialsError:
                logger.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
                raise
        return self._client
    
    @property
    def resource(self):
        """
        Lazy initialization of S3 resource.
        """
        if self._resource is None:
            try:
                self._resource = boto3.resource(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region
                )
            except NoCredentialsError:
                logger.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
                raise
        return self._resource
    
    def download_file(self, 
                     key: str, 
                     local_path: str, 
                     bucket_name: Optional[str] = None) -> bool:
        """
        Download a file from S3 to a local path.
        
        Args:
            key (str): The S3 object key (file path in S3)
            local_path (str): Local file path where the file should be saved
            bucket_name (str, optional): S3 bucket name. If not provided, uses default bucket.
            
        Returns:
            bool: True if download was successful, False otherwise
            
        Example:
            s3_service = S3Service()
            success = s3_service.download_file('documents/file.pdf', '/tmp/downloaded_file.pdf')
        """
        bucket = bucket_name or self.aws_s3_bucket
        
        if not bucket:
            logger.error("No bucket specified and no default bucket configured.")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            self.client.download_file(bucket, key, local_path)
            logger.info(f"Successfully downloaded {key} from {bucket} to {local_path}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File {key} not found in bucket {bucket}")
            elif error_code == 'NoSuchBucket':
                logger.error(f"Bucket {bucket} not found")
            else:
                logger.error(f"Error downloading file {key}: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error downloading file {key}: {e}")
            return False
    
    def download_file_to_memory(self, 
                              key: str, 
                              bucket_name: Optional[str] = None) -> Optional[bytes]:
        """
        Download a file from S3 to memory.
        
        Args:
            key (str): The S3 object key (file path in S3)
            bucket_name (str, optional): S3 bucket name. If not provided, uses default bucket.
            
        Returns:
            bytes: File content as bytes if successful, None otherwise
            
        Example:
            s3_service = S3Service()
            content = s3_service.download_file_to_memory('documents/file.pdf')
        """
        bucket = os.getenv("AWS_S3_BUCKET")
        
        if not bucket:
            logger.error("No bucket specified and no default bucket configured.")
            return None
        
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            logger.info(f"Successfully downloaded {key} from {bucket} to memory")
            return content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File {key} not found in bucket {bucket}")
            elif error_code == 'NoSuchBucket':
                logger.error(f"Bucket {bucket} not found")
            else:
                logger.error(f"Error downloading file {key}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error downloading file {key}: {e}")
            return None
    
    def get_file_metadata(self, 
                         key: str, 
                         bucket_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a file in S3 without downloading it.
        
        Args:
            key (str): The S3 object key (file path in S3)
            bucket_name (str, optional): S3 bucket name. If not provided, uses default bucket.
            
        Returns:
            dict: File metadata if successful, None otherwise
            
        Example:
            s3_service = S3Service()
            metadata = s3_service.get_file_metadata('documents/file.pdf')
        """
        bucket = bucket_name or self.aws_s3_bucket
        
        if not bucket:
            logger.error("No bucket specified and no default bucket configured.")
            return None
        
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            metadata = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType'),
                'etag': response['ETag'].strip('"'),
                'metadata': response.get('Metadata', {})
            }
            logger.info(f"Successfully retrieved metadata for {key} from {bucket}")
            return metadata
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File {key} not found in bucket {bucket}")
            elif error_code == 'NoSuchBucket':
                logger.error(f"Bucket {bucket} not found")
            else:
                logger.error(f"Error getting metadata for file {key}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error getting metadata for file {key}: {e}")
            return None
    
    def file_exists(self, 
                   key: str, 
                   bucket_name: Optional[str] = None) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            key (str): The S3 object key (file path in S3)
            bucket_name (str, optional): S3 bucket name. If not provided, uses default bucket.
            
        Returns:
            bool: True if file exists, False otherwise
            
        Example:
            s3_service = S3Service()
            exists = s3_service.file_exists('documents/file.pdf')
        """
        bucket = bucket_name or self.aws_s3_bucket
        
        if not bucket:
            logger.error("No bucket specified and no default bucket configured.")
            return False
        
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return False
            else:
                logger.error(f"Error checking if file {key} exists: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error checking if file {key} exists: {e}")
            return False
    
    def generate_presigned_download_url(self, 
                                      key: str, 
                                      expiration: int = 3600,
                                      bucket_name: Optional[str] = None) -> Optional[str]:
        """
        Generate a presigned URL for downloading a file from S3.
        
        Args:
            key (str): The S3 object key (file path in S3)
            expiration (int): Time in seconds for the presigned URL to remain valid (default: 1 hour)
            bucket_name (str, optional): S3 bucket name. If not provided, uses default bucket.
            
        Returns:
            str: Presigned URL if successful, None otherwise
            
        Example:
            s3_service = S3Service()
            url = s3_service.generate_presigned_download_url('documents/file.pdf', expiration=7200)
        """
        bucket = bucket_name or self.aws_s3_bucket
        
        if not bucket:
            logger.error("No bucket specified and no default bucket configured.")
            return None
        
        try:
            response = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            logger.info(f"Successfully generated presigned URL for {key} from {bucket}")
            return response
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL for file {key}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL for file {key}: {e}")
            return None


# Convenience function to get a singleton instance
_s3_service_instance = None

def get_s3_service() -> S3Service:
    """
    Get a singleton instance of S3Service.
    
    Returns:
        S3Service: Singleton instance of the S3 service
    """
    global _s3_service_instance
    if _s3_service_instance is None:
        _s3_service_instance = S3Service()
    return _s3_service_instance
