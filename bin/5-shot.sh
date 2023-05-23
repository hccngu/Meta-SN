# huffpost
dataset=huffpost
data_path="../data/huffpost.json"
n_train_class=20
n_val_class=5
n_test_class=16
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --train_epochs 3000 \
    --test_epochs 1000 \
    --val_epochs 200 \
    --train_iter 18 \
    --test_iter 12 \
    --meta_lr 1e-5 \
    --task_lr 5e-1 \
    --Comments "huffpost " \
    --patience 20 \
    --seed 42 \
    --notqdm \
    --weight_decay 1e-5 \
    --dropout 0.0 \
    --train_loss_weight 10.0 \
    --test_loss_weight 5.8 \
    --kernel_size 2 3 5 \

# amazon
dataset=amazon
data_path="../data/amazon.json"
n_train_class=10
n_val_class=5
n_test_class=9
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --train_epochs 3000 \
    --test_epochs 1000 \
    --val_epochs 200 \
    --train_iter 18 \
    --test_iter 30 \
    --meta_lr 2e-5 \
    --task_lr 4e-1 \
    --Comments "amazon " \
    --patience 20 \
    --seed 42 \
    --notqdm \
    --weight_decay 1e-5 \
    --dropout 0.0 \
    --train_loss_weight 12.0 \
    --test_loss_weight 5.0 \
    --kernel_size 2 3 4 \

# 20newsgroup
dataset=20newsgroup
data_path="../data/20news.json"
n_train_class=8
n_val_class=5
n_test_class=7
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --train_epochs 3000 \
    --test_epochs 1000 \
    --val_epochs 200 \
    --train_iter 18 \
    --test_iter 15 \
    --meta_lr 1e-5 \
    --task_lr 5e-1 \
    --Comments "20newsgroup " \
    --patience 20 \
    --seed 42 \
    --notqdm \
    --weight_decay 1e-5 \
    --dropout 0.0 \
    --train_loss_weight 12.0 \
    --test_loss_weight 5.8 \
    --kernel_size 2 3 4 \

# reuters
dataset=reuters
data_path="../data/reuters.json"
n_train_class=15
n_val_class=5
n_test_class=11
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --train_epochs 3000 \
    --test_epochs 1000 \
    --val_epochs 200 \
    --train_iter 18 \
    --test_iter 30 \
    --meta_lr 2e-5 \
    --task_lr 2e-1 \
    --Comments "reuters " \
    --patience 20 \
    --seed 3 \
    --notqdm \
    --weight_decay 1e-5 \
    --dropout 0.0 \
    --train_loss_weight 10.0 \
    --test_loss_weight 5.8 \
    --kernel_size 2 3 5 \
    
# fewrel
dataset=fewrel
data_path="../data/fewrel.json"
n_train_class=65
n_val_class=5
n_test_class=10
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --train_epochs 3000 \
    --test_epochs 1000 \
    --val_epochs 200 \
    --train_iter 18 \
    --test_iter 12 \
    --meta_lr 2e-5 \
    --task_lr 5e-1 \
    --Comments "fewrel " \
    --patience 20 \
    --seed 42 \
    --notqdm \
    --weight_decay 1e-5 \
    --dropout 0.0 \
    --train_loss_weight 12.0 \
    --test_loss_weight 5.8 \
    --kernel_size 2 3 4 \
    
# rcv1
dataset=rcv1
data_path="../data/rcv1.json"
n_train_class=37
n_val_class=10
n_test_class=24
python ../src/main_simaese_network.py \
    --cuda 0 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --embedding mlada \
    --classifier r2d2 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --train_episodes 3 \
    --train_epochs 3000 \
    --test_epochs 1000 \
    --val_epochs 200 \
    --train_iter 18 \
    --test_iter 12 \
    --meta_lr 5e-5 \
    --task_lr 2e-1 \
    --Comments "rcv1 " \
    --patience 20 \
    --seed 42 \
    --notqdm \
    --weight_decay 1e-5 \
    --dropout 0.0 \
    --train_loss_weight 12.0 \
    --test_loss_weight 5.8 \
    --kernel_size 2 3 4 \
    