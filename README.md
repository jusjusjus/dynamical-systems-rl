
# Lorenz Environment

## Version 0

* Observable.  `X = (x, y, z)` at each time step.
* Reward.  `r = (dt*(0.5+0.5*(action==_NO_ACTION))) if x>0 else -dt`
* Action.  `action=0, .., 3`.  For `i=0, 1, and 2`, a constant force with strength 30 is applied to variable `X[i]`.  For `i=3` nothing is done.
