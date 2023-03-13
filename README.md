## Training
Sample trajectory $\tau$ from an epoch $\epsilon$ with probability $P_{\epsilon} = \frac{r(\epsilon)}{\sum_\epsilon{r(\epsilon)}}$ with randomly selected beginning and ending steps  
For each step n in $\tau$: $\hat S_n, \hat A_n, \hat R_n = \pi(\tau_{0: n - 1})$  
$J_\theta(\tau) = -\log(\pi(\tau) * R)$
