# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Code for the Federated Learning project.

## Setup
#### Environment
The project is designed to run on google colab, run it be sure to follow this steps:
1. Install required packages:
   ```
   pip install wandb
   pip install overrides
   ```
2. Add Idda dataset in ```data/idda```
3. Add GTA dataset in ```data/gta```
4. If wandb must be used to save metrics and models login using the command ```wandb login``` (otherwise add the argument ```--not_use_wandb``` on the run)

#### Datasets
The repository supports experiments on the following datasets: 
1. Reduced **Federated IDDA** from FedDrive [1]
   - Task: semantic segmentation for autonomous driving
   - 24 users
2. Reduced **GTA**
   - Task: semantic segmentation for autonomous driving
   - 20 classes

## How to run
The ```run.py``` orchestrates everything. All arguments need to be specified through the ```args``` parameter (options can be found in ```config/args.py```).
Example of experiment:
```bash
python run.py
  --project federated-training 
  --exp_name big-round 
  --seed 0 
  --training_ds idda
  --test_ds idda
  --model deeplabv3_mobilenetv2 
  --num_epochs 5 
  --bs 4 
  --optimizer SGD 
  --lr 0.1 
  --weight_decay 0 
  --momentum 0.9 
  --lr_policy poly 
  --lr_power 0.9 
  --lr_decay_step 15 
  --lr_decay_factor 0.1 
  --rrc_transform 
  --min_scale 0.5 
  --max_scale 2.0 
  --h_resize 756 
  --w_resize 1344 
  --norm eros_norm 
  --jitter 
  --phase all
  --not_use_wandb 
  federated 
  --num_rounds 500 
  --clients_per_round 5
```

## How to test best ClAvBN and SiloBN model
1. **ClAvBN**: run the file ```clavbn.sh```
1. **SiloBN**: run the file ```silobn.sh```

## References
[1] Fantauzzo, Lidia, et al. "FedDrive: generalizing federated learning to semantic segmentation in autonomous driving." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.
