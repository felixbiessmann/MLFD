import os
import boto3

s3_client = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')
s3_client._request_signer.sign = (lambda *args, **kwargs: None)


def download_dir(prefix, local, bucket, client=s3_client):
    """
    Downloads a directory from s3. Taken from the good people
    over at stackoverflow: https://stackoverflow.com/a/56267603

    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': bucket,
        'Prefix': prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)


folder_name = 'MLFD_fd_detection'
bucket_name = 'mlfd-data'
try:
    os.mkdir(folder_name)
    print('Downloading. This may take a while...')
    download_dir('', folder_name, bucket_name, s3_client)
    print('Finished fetching binaries')
except FileExistsError:
    print(f'The {folder_name} directory already exists. Delete it to\
            re-download the binaries.')
