import numpy as np


class LinearLR():
    def __init__(self, 
        optimizer,
        lr_start=0.0, 
        lr_peak=1e-4, 
        lr_end=1e-5,
        warmup_steps=1000,
        total_steps=10000):

        self.opt = optimizer
        self.steps_warmup
