# Point-Mass RL Example

This directory is self-contained:

- `pointmass_rl.ipynb` is the public notebook.
- `pointmass_rl.py` is the script form of the same training run.
- `policy.py` and `optimizer.py` are the small local helpers used by both.
- `outputs/` contains the recorded checkpoint, learning curve, and figures.

The notebook loads the recorded outputs by default. Set `RETRAIN = True`
inside the notebook to regenerate the checkpoint and learning-curve CSV.
