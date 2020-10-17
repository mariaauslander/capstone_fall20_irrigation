# INSTRUCTIONS FOR CLOUD SETUP
## Tested Only on IBM Cloud with S3Fuse

Follow the steps below for setting up the appropriate environment for model training.

1. Requisition a GPU with > 1TB additional mounted disk. Ideally a V100 as it trains 3x faster than a P100.
2. ssh to this GPU 
3. Install docker 
```
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
	
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"	
apt-get update
apt-get install -y docker-ce
# verify
docker run hello-world
```
4. Mount the additional disk
```
fdisk -l
mkdir -m 777 /data
mkfs.ext4 /dev/xvdc

# edit /etc/fstab and all this line:
/dev/xvdc /data                   ext4    defaults,noatime        0 0

mount /data
```
5. Move docker files to the mounted disk or they will fill up the working node
```
service docker stop
cd /var/lib
cp -r docker /data
rm -fr docker
ln -s /data/docker ./docker
service docker start
```

6. Connect to your cloud storage with the MSI data using S3Fuse. You can use the commands below to install S3Fuse on an IBM Cloud device.
```
sudo apt-get update
sudo apt-get install -y automake autotools-dev g++ git libcurl4-openssl-dev libfuse-dev libssl-dev libxml2-dev make pkg-config
git clone https://github.com/s3fs-fuse/s3fs-fuse.git
ls
cd s3fs-fuse
./autogen.sh
./configure
make
sudo make install

Substitue your values for <Access_Key_ID> and <Secret_Access_Key> in the below command.
echo "<Access_Key_ID>:<Secret_Access_Key>" > $HOME/.cos_creds
chmod 600 $HOME/.cos_creds

pip install --upgrade s3cmd

# Sub your access keys in the config file below
cat >.s3cfg << eof
 [default]

access_key = <Access_Key_ID>
secret_key = <Secret_Access_Key>
gpg_command = /usr/local/bin/gpg
# host_base = s3.private.us-south.cloud-object-storage.appdomain.cloud
# host_bucket = %(bucket)s.s3.private.us-south.cloud-object-storage.appdomain.cloud
host_base =s3.private.us-east.cloud-object-storage.appdomain.cloud
host_bucket = %(bucket)ss3.private.us-east.cloud-object-storage.appdomain.cloud
use_https = True
eof

# Finally let's make our mounted directory
sudo mkdir -m 777 /mnt/irrigation_data
sudo s3fs irrigation.data /mnt/irrigation_data -o passwd_file=$HOME/.cos_creds -o sigv2 -o use_path_request_style -o url=https://s3.us-east.objectstorage.softlayer.net

```
7. Clone this GitHub repo
8. Navigate to this repo `cd capstone_fall20_irrigation`
9. Build the docker image using the command:  `docker build -t irgapp -f ./setup/tf23.docker .`
