# Data driven scene simulation
This folder contains the code for endoscopy scene reconstruction and simulation.

## Installation
```
pip install -r requirements.txt
pip install submodules/surrol_gs
pip install submodules/simple-knn
```

## Running
```
cd surrol/tasks
python soft_retraction.py
python gs_interaction.py
```

### Reconstruction
Please refer to [reconstruction](./reconstruction/README.md) to configure the environment and reconstruct the scene from endoscopy dataset.