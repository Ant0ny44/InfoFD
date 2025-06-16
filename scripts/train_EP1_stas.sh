K=7 # K random seeds.

for ((i=0; i<K; i++)); do
    echo "Running experiment with random seed  $RANDOM"
    python train.py --config_path ./configs/EP1.yml --device cuda:0  --exp_name GenImage-EP1-stat --tags  GenImage EP1 dto stat --save_pth --seed  $RANDOM
done 