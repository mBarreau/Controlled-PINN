# Controlled PINN

This repository contains the code for the paper [A Control Perspective on Training PINNs](https://arxiv.org/abs/2501.18582). 

The main contribution is a formulation of the training of a PINN using a gradient-descent scheme as a dynamical system. The resampling subroutine leads to unbiased noise and disturbances in the process which can be effectively mittigated using an integral controller. A robust extension with a leaky integrator is also proposed as an anti windup strategy.

This code evaluates the performance of the algorithms on a toy example.

For the installation, use Python 3.12 and install the necessary packages in `requirements.txt`. The example can be ran from `example.py` and creates csv files in the `output` folder. The analysis is conducted in `analysis.py` creating tables in the console and an image in the same folder.

