# Endoscopy Reconstruction

## Installation
```
pip install -r requirements.txt
pip install submodules/gs-render
pip install submodules/simple-knn
```

## Running
Write your configs file at first. We have provided a config file [traction.yaml](./configs/traction.yaml) in configs folder.
```
python train.py --config ./configs/traction.yaml
```