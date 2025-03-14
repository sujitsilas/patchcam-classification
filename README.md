
# Creating a python virtual environment on [Hoffman2](https://www.hoffman2.idre.ucla.edu/)

## Setup Instructions

#### 2. Load the python module (adjust the module name/version as necessary)
```bash
module load python/3.7.3 
```
#### 3. Create a python environment
```bash
python -m venv /path/to/envs/myenv
```
#### 4. Install the required packages
```bash
source myenv/bin/activate
pip install --user matplotlib pandas torch torchvision tqdm albumentations Pillow opencv-python scikit-learn h5py efficientnet_pytorch torchstain

```
#### 5. Deactivate the python environment
```bash
deactivate
```

### Request an interactive node and try running an executable bash script
#### cuda represent the number of GPUs you want. Can request up to 4 v100s on hoffman
```bash
qrsh -l gpu,A100,cuda=1,h_rt=2:00:00
```

##### Interactive session with CPUs (if needed)
```bash
qrsh -l h_data=15G,h_rt=3:00:00,h_vmem=4G -pe shared 4
```


