# PLCR

This is the implemention of our ICML'22 paper (Revisiting **C**onsistency **R**egularization for Deep **P**artial **L**abel Learning).

Requirements: 
Python 3.6.9, 
numpy 1.19.5, 
torch 1.9.1,
torchvision 0.10.1.

You need to:
1. Download SVHN and CIFAR-10 datasets into './data/'.
2. Run the following demos:
```
python main.py --dataset cifar10 --model widenet --data-dir ./data/cifar10/ --lam 1 --lr 0.1 --trial 1  --rate 0.7
python main.py --dataset cifar10 --model widenet --data-dir ./data/cifar10/ --lam 1 --lr 0.1 --trial 1  --rate=-1
python main.py --dataset cifar100 --model widenet --data-dir ./data/cifar100/ --lam 1 --lr 0.1 --trial 1  --rate 0.1
```

If you have any further questions, please feel free to send an e-mail to: dongdongwu@seu.edu.cn. Have fun!
