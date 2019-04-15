# MachineLearning

This repository draws attention to simple machine learning algorithms such as Linear Regression, Logistic Regression, Gradient Descent and Principal Component Analysis. The two latter algorithms can be found in the "algorithms.py" file, whereas the two former models are implemented in the files "linear_model.py" and "non_linear_model.py". Loss functions and evaluation metrices can be found in the respective files. 

Since mean centering (subtracting the mean from each data point $x_i$) and normalization (in addition to subtracting the mean, dividing by the standard deviation) is an important pre-processing step (if features happen to be on a different scale), a scaler object can be found in the file "feature_caling.py", which either computes mean centering or (additionally) normalization - dependent on the set-up. 

For each of the machine learning algorithms I created a distinct class, making use of Python's neat object oriented programming features.
