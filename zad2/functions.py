#!/usr/bin/env python3

import numpy as np

def sigmoid(x: float) -> float:
    return(1/(1 +np.exp(-x)))

def derivsig(x: float) -> float:
    return(sigmoid(x) * (1 - sigmoid(x)))
