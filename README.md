# StudentChangePoint
This code was released for [our paper "Detecting Changes of Student Behavior from Clickstream Data" at LAK 17](http://dl.acm.org/citation.cfm?id=3027430)

## student_changepoint.py
This is the main file for student change detection.
It loads a matrix with size (`n_students X n_days`) in csv format and runs change detection analysis. 
Take a look the main function at the bottom for an example run (or you can just run this file).

## glm_gd.py
Generalized linear regression (that only works for our model) using simple gradient descent.
It does not seem to be working well with the Poisson model. The code is only used for the Bernoulli model.

## utils.py
Util functions.
