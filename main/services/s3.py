import os
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
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
        
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")
        self.aws_s3_bucket = os.getenv("AWS_S3_BUCKET")
        
        self._client = None
    
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
        
        if not self.aws_s3_bucket:
            logger.error("No bucket specified and no default bucket configured.")
            return None
        
        try:
            response = self.client.get_object(Bucket=self.aws_s3_bucket, Key=key)
            content = response['Body'].read()
            logger.info(f"Successfully downloaded {key} from {self.aws_s3_bucket} to memory")
            return content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File {key} not found in bucket {self.aws_s3_bucket}")
            elif error_code == 'NoSuchBucket':
                logger.error(f"Bucket {self.aws_s3_bucket} not found")
            else:
                logger.error(f"Error downloading file {key}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error downloading file {key}: {e}")
            return None
