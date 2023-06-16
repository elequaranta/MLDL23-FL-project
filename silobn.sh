git clone https://github.com/SamueleVanini/MLDL23-FL-project.git

# Import packages needed in colab
pip install wandb
pip install overrides

# Login in wandb
wandb login dfb691367550ee2d3f2c3a4f4849e10d065188fb

# Move in the correct folder
cd MLDL23-FL-project 

python run.py \
  --project exam-project \
  --exp_name no_fda_basic_silo_up_10_cl_4 \
  --load_checkpoint fed-server-silo-self-learning-100.torch mldlproject/basic-silo-self-learning/089yp65x \
  --seed 0 \
  --training_ds idda_silo \
  --test_ds idda_silo \
  --model deeplabv3_mobilenetv2 \
  --num_epochs 1 \
  --bs 4 \
  --optimizer SGD \
  --lr 0.01 \
  --weight_decay 0 \
  --momentum 0.9 \
  --lr_policy poly \
  --lr_power 0.9 \
  --lr_decay_step 15 \
  --lr_decay_factor 0.1 \
  --rrc_transform \
  --min_scale 0.5 \
  --max_scale 2.0 \
  --h_resize 512 \
  --w_resize 928 \
  --norm eros_norm \
  --phase test \
  basic_silo \
  --num_rounds 100 \
  --clients_per_round 4 \
  --update_teach 10 \
  --conf_threshold 0.9 \
  --alpha 0.3 \
  --beta 0.7 \
  --tau 2