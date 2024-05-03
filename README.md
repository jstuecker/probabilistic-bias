# probabilistic-bias
This repository contains some code to compute symbolic expressions for the probabilistic estimators of tensorial bias, as explained in Section 4 of the article "Probabilistic Lagrangian bias estimators and the cumulant bias
expansion" (Stuecker et al., 2023)

If you have no clue what this means, then you are probably in the wrong place, since this is some highly specific code without any other applications.

This repository contains the following files:
* [isotropic_tensors.py](isotropic_tensors.py): This is the python module that is used to create numerical representations of the tensors, to symmetrize and orthogonalize them and to compute their algebra
* [tensor_algebra.ipynb](tensor_algebra.ipynb): Shows some examples how to evaluate products between different tensors and how to obtain e.g. the covariance matrix of the tidal tensor and its (generalized) inverse. This notebook is a good place to start.
* [tensor_estimators.ipynb](tensor_estimators.ipynb): Shows how to use the tensor algebra to compute the bias estimators, e.g. for the tidal bias and other higher order terms
* [bias_estimators.py](bias_estimators.py): This file is created automatically by [tensor_estimators.ipynb](tensor_estimators.ipynb) so you might want to look at that notebook first

# Requirements
* sympy
* numpy

# Further
This code represents only a small slice of the code that is necessary to evaluate all the bias estimators in the paper. I might upload the remaining code here in the future. If you are interested in this, please drop me a mail, so that this wanders a bit higher in priority on my ToDo list!
