# package_hp_opti

Hyper-parameter optimization.

### random search 
- example with gradient boosted regressor and a scoring = variance * correlation / mse.
- N.B: the code isn't fully generic, you have to change the functions in order to apply it with other models or other scores.

### grid search 
- simple grid search and grid search with kbest feature selection

### Bayesian optimization
- example with gradient boosted regressor and a scoring = variance * correlation / mse.
- N.B: the code isn't fully generic, you have to change the functions in order to apply it with other models or other scores.

### References
- Bayesian optimization : https://ailab.criteo.com/hyper-parameter-optimization-algorithms-a-short-review/
- Hyperopt : https://blog.dominodatalab.com/hyperopt-bayesian-hyperparameter-optimization/ 
