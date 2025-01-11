def ensure_bucket_exists(s3_client, bucket_name):
    """Create bucket if it doesn't exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Created bucket: {bucket_name}")
        except Exception as e:
            print(f"Error creating bucket: {str(e)}")
            raise