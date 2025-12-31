# AdaptedMoE V0.9 2025.12.31
![overview](imgs/overview.png)
## Project Homepage
https://ray3572.github.io/AdaptedMoE_web/

## Features
1. optimize train/test process of   
SimpleNet:   
https://github.com/DonaldRR/SimpleNet  
**3X faster than the original code with same precision**  
2. AdaptedMoE

## Install  
conda create -n AdaptedMoE python=3.10  
conda activate AdaptedMoE  
pip install -r requirements.txt  

## Config
modify model parameters in config/  
self.data_path="Your data root of mvtec"  
self.datasets="type of train target"

## Run
python train_AdaptedMoE.py  
python train_SimpleNet_baseline.py  

## Reference
https://github.com/DonaldRR/SimpleNet  




