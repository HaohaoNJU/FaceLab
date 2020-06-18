# For face recognition and other metric learning tasks 
A easy framework for training and validation, the training pipline has been seperate several parts(networks,metrics & loss functions, data loaders) into different folders accordingly. moreover, in order to be extensible, 
major training modules are written in `modules.py` which includes:
* `IOFactory` for logging, reading and saving.
* `OptimFactory` for params optimization and lr scheduling
* `Header` for pairwise metric learning or metrics with classification

The overall training protocol has been written in file solver.py, while the configurations are set in file config.yaml, before running `python solver.py --cfg config.yaml`
to train a model, you may just feel free to change and config you own. 

Experiment shows a consistent result with InsightFace, validation and some other modules are to be added!

# Compatibility
The code has been tested using Pytorch r1.2.0 under Ubuntu16 with python3.6

>>>>>>> v1.0
