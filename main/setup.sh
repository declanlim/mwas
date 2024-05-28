# install required software
sudo yum install git python pip gcc python3-devel parallel

# install aws cli
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  #unzip awscliv2.zip
  #sudo ./aws/install
unzip awscliv2.zip
rm awscliv2.zip
sudo ./aws/install

# must be done by hand?
aws configure
aws configure set default.s3.max_concurrent_requests 300  # 300 is too much??? for a small (RAM) machine


python -m pip install --upgrade pip
pip install numpy pandas scipy boto3 psycopg2-binary s3fs fsspec

# get mwas files and copy to directory
git clone https://github.com/declanlim/mwas_rfam.git
cp mwas_rfam/mwas_rfam.py .
cp -r mwas_rfam/family_groups .
rm -rf mwas_rfam

# create logging folder
mkdir logs
