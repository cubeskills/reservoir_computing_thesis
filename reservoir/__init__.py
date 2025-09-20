# In __init__.py
from reservoir.core.reservoir import Reservoir
from reservoir.core.input_mapping import InputMapping
from reservoir.core.readout import Readout, LinearReadout, RidgeReadout
from reservoir.training.trainer import ReservoirTrainer
from reservoir.utils.metrics import mse, rmse, nrmse