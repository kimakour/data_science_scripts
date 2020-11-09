# data_science_scripts
The project contains useful functions to build data pipelines, statistical metrics for data analysis and some visualizations tools.
It is divided into the following parts :
### dashboard
An example of a dashboard build by using Dash from Plotly for the fish dataset. 
It contains simple visualizations like histograms, scatter plots, heatmaps.
![alt text](https://github.com/kimakour/data_science_scripts/blob/master/images/dash_1.png)
It contains also some modeling performances like a pie chart for the absolute error, a scatter for the error accoring to the prediction and a REC curve with AUC score.
![alt text](https://github.com/kimakour/data_science_scripts/blob/master/images/dash_2.png)
Finally it contains some model interpretability using SHAP values.
![alt text](https://github.com/kimakour/data_science_scripts/blob/master/images/dash_3.png)

### distribution_distance
A Python package to calculate KL divergence, PSI values and group entities with the same distribution probabilities. 
A jupyter notebook is provided in the folder jupyter_notebooks for an example of using PSI values and group entities within a certain threshold.
![alt text](https://github.com/kimakour/data_science_scripts/blob/working_branch/images/psi_group.png)

### jupyter_notebooks

A list of notebooks including different modeling techniques, like ARIMA models , clustering for mixed data like K prototypes, mixed data distances like Gower distance and so on ...

### package_hp_opti
A Python package to do hyper-parameter optimization : random search , grid search and bayesian optimization.

### package_interpretability 
A Python package to interpret samples for anomaly detection with autoencoders and calculate weighted similarities (based on feature importance) for samples.

### package_model_comparing
A Python package for a pipeline to compare between classification models and regression models.

### package_plot 
A Python package containing simple plots.

### package_processing
A Python package to do some transformations, like polynomial transformation, power transformation, IQR ...
