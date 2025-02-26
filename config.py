import os
import boto3

redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
queue_name = os.getenv("QUEUE_NAME")

s3_access_key = os.getenv("AWS_S3_ACCESS_KEY")
s3_secret_key = os.getenv("AWS_S3_SECRET_KEY")
s3_bucket = os.getenv("AWS_S3_BUCKET")
s3_region = os.getenv("AWS_REGION")

channel_name = os.getenv("CHANNEL_NAME")

s3_client = boto3.client(
    's3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    region_name=s3_region
)


