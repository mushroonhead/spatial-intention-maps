#!/bin/sh

# run this file in a directory for this project
# for example, we are currently running this in `/home/joshmah/spatial_int_map_diffuser`

git clone https://github.com/mushroonhead/spatial-intention-maps.git

module purge
module load python/3.10.4
module load cuda/11.8.0

mkdir envs/
cd envs/
python -m venv spatial_int_maps_3_10_4
source spatial_int_maps_3_10_4/bin/activate

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install pybullet tensorboard future scipy scikit-image pyyaml munch pandas tqdm prompt-toolkit Cython opencv-python pyqt5 fpdf matplotlib
# pip install anki-vector zeroconf opencv-contrib-python pyglet
cd ../spatial-intention-maps/shortest_paths
python setup.py build_ext --inplace

deactivate
module purge