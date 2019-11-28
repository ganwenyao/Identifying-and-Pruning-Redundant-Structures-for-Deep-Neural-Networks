# Identifying and Pruning Redundant Structures for Deep Neural Networks

## Dependencies
torch v1.0.1, torchvision

## Train baseline model

```shell
python3 main.py --dataset cifar10 --arch vgg --depth 16 --save save/vgg/main
python3 main.py --dataset cifar10 --arch preresnet --depth 56 --save save/preresnet/main
```

## Step 1: Normal training on the training set

```shell
python3 main.py --dataset cifar10 --arch vgg --depth 16 --valid --save save/vgg/main_val1/
python3 main.py --dataset cifar10 --arch preresnet --depth 56 --valid --save save/preresnet/main_val0/
```

## Step 2: Iterative pruning and fine-tuning (IPFT)

```shell
python3 vggprune_layer.py --dataset cifar10 --epochs 1 --lr 0.1  --pruneT 0.91 --dropT 0.01 --valid --model save/vgg/main_val1/model_best.pth.tar --save save/vgg/vggprune_layer/0.91
python3 vggprune_filter.py --dataset cifar10 --epochs 1 --lr 0.1  --pruneT 0.91 --dropT 0.005 --min-filters 8  --valid --model save/vgg/vggprune_layer/0.91/checkpoint.pth.tar.end --save save/vgg/vggprune_filter/0.91

python3 preresnetprune_block.py --dataset cifar10 --epochs 1 --lr 0.001  --pruneT 0.92 --dropT 0.005 --valid --model save/preresnet/main_val0/model_best.pth.tar --save save/preresnet/preresnetprune_block/0.92 
python3 preresnetprune_filter.py --dataset cifar10 --epochs 1 --lr 0.1  --pruneT 0.92 --dropT 0.0025 --min-filters 8 --valid --reverse --model save/preresnet/preresnetprune_block/0.92/checkpoint.pth.tar.end --save save/preresnet/preresnetprune_filter/0.92  
```

## Step 3: Normal training on both the training set and validation set

```shell
python3 main_cfg.py --dataset cifar10 --arch vgg --resume save/vgg/vggprune_filter/0.91/checkpoint.pth.tar.end --save save/vgg/retrain/0.91
python3 main_cfg.py --dataset cifar10 --arch preresnet --resume save/preresnet/preresnetprune_filter/0.92/checkpoint.pth.tar.end --save save/preresnet/retrain/0.92
```