# Welcome to Atari Games
***

## Task
TODO - What is the problem? And where is the challenge?

## Description
TODO - How have you solved the problem?

## Installation
TODO - How to install your project? npm install? make? make re?

## Usage
TODO - How does it work?
```
./my_project argument1 argument2
```

### Fast training on M1 (CPU recommended)

These configs apply the faster DQN settings and run on CPU, which benchmarks show
is faster than MPS for this project.

```
. .venv/bin/activate
python train/train_cartpole.py --config configs/cartpole_m1_fast.json
python train/train_atari.py --config configs/space_invaders_m1_fast.json
python train/train_atari.py --config configs/pacman_m1_fast.json
```

Turbo (NoisyNet + update-every-frame + strong early stop targets):

```
. .venv/bin/activate
python train/train_cartpole.py --config configs/cartpole_m1_turbo.json
python train/train_atari.py --config configs/space_invaders_m1_turbo.json
python train/train_atari.py --config configs/pacman_m1_turbo.json
```


### The Core Team


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px' /></span>
