set -e

# Redice size
pip uninstall torch -y

# Create env
conda create -n purple python=3.9 
source activate purple
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
