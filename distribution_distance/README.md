# distribution_distance
The package measures how much a variable has shifted in distribution between two samples or over time. 
contains two files :
### kl_divergence.py
- compute KL divergence 
 <img src="https://latex.codecogs.com/gif.latex?D_{KL}(P||Q)=\sum_{i}P(i)*\ln(\frac{P(i)}{Q(i))})" /> 
 
- compute Jensen-Shannon distance 
<img src="https://latex.codecogs.com/gif.latex?M=\frac{1}{2}(P+Q)" />
<img src="https://latex.codecogs.com/gif.latex?D_{JensenShannon}(P,Q)=\sqrt{\frac{1}{2}*(D(P||M)+D(Q||M))}" />

- compute Population stability index (PSI) value 
<img src="https://latex.codecogs.com/gif.latex?PSI(P,Q)=D_{KL}(P||Q)+D_{KL}(Q||P)" />

-plot PSI values over a dataset


### psi_grouping.py

- select a categorical feature and creates clusters of its levels according to the disribution distance of the other features.


### References 
- KL divergence : https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- Jensen–Shannon divergence : https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
- Population Stability Index : https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations
