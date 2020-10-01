mkdir BigEarthData
mkdir BigEarthData/models
mkdir BigEarthData/tfrecords
ls -l BigEarthData

apt install python3-pip

pip3 install python
pip3 install --upgrade pip

apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install vim
apt install python3.7

alias python='python3.7'

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install pandas
pip install tensorflow
pip install rasterio
pip install matplotlib
pip install tqdm
pip install seaborn
