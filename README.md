
# Dynamical Systems RL

Reinforcement learning with dynamical systems.

## Introduction

To install run

```bash
$ conda create -n dyrl python=3.6 pip
$ conda activate dyrl
$ pip install -r requirements.txt
```

To tests run

```bash
$ pytest
```

To try it out run

```bash
$ python ./scripts/train-qmodel-lorenz.py
```

# Lorenz Environment

## Version 0

* Observable.  `X = (x, y, z)` at each time step.
* Reward.  `r = (dt*(0.5+0.5*(action==_NO_ACTION))) if x>0 else -dt`
* Action.  `action=0, .., 3`.  For `i=0, 1, and 2`, a constant force with strength 30 is applied to variable `X[i]`.  For `i=3` nothing is done.
