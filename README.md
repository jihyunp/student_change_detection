# StudentChangepoint
This code was released for [our paper "Detecting Changes of Student Behavior from Clickstream Data" at LAK 17](http://dl.acm.org/citation.cfm?id=3027430)

Jihyun Park (`jihyunp@ics.uci.edu`)<br>
March 2017

## Required Packages
Written in `python2.7`. <br>
Python packages `numpy`, `matplotlib`, `statsmodels` are needed.


## Code
### student_changepoint.py
This is the main file for student change detection.
It loads a matrix with size (`n_students X n_days`) in csv format and runs the changepoint detection.
Take a look the main function at the bottom for an example run (or you can just run this file).

* __Class `StudentChangepoint`__<br>
`detected_cp_arr`: detected changepoint locations<br>
`mcp_max_ll_mat`, `m0_ll_mat` : LogLik values for model w/ cp and model w/o cp<br>
`mcp_min_bic_mat`, `m0_bic_mat` : BIC values for model w/ cp and model w/o cp<br>
`alpha_i_mat` : Three column matrix for alpha_i's. `[alpha_i1, alpha_i2, alpha_i0]`<br>
`better_w_cp_sidxs` : Student indices with detected change<br>
`better_wo_cp_sidxs` : Student indices without detected change<br>

### glm_gd.py
Generalized linear regression (that only works for our model) using a simple gradient descent method.
It does not seem to be working well with Poisson model. The code is only used for the Bernoulli model.

### utils.py
Some useful functions that are being used.

