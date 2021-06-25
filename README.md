# Stage Ruptures


## How to use Alpin

```python
from src.alpin import Alpin

list_of_signals=[signal_1, signal_2,...]
list_of_bkps=[bkps_1, bkps_2,...]


estimator = rpt.Pelt(model="l2")

algo = Alpin(estimator=estimator).fit(list_of_signals, list_of_bkps)

print(f"Best penalty value: {algo.get_best_penalty(signal_1)}")
print(f"Best penalty weights: {algo.best_penalty_weights}")

# prediction with the best found parameters
my_bkps = algo.predict(signal=signal_1)
```

By default, the only signal feature taken into account is `log(n_samples)` where `n_samples` is the signal length.
To use more features, for instance the variance, do:

```python
import numpy as np

def feature_funct(signal: np.ndarray)->np.ndarray:
    # the last element (2.) is simply a constant.
    return np.array([np.log(signal.shape[0]), signal.var(), 2.])

algo = Alpin(estimator=estimator, feature_func=feature_func).fit(list_of_signals, list_of_bkps)

print(f"Best penalty value: {algo.get_best_penalty(signal_1)}")
print(f"Best penalty weights: {algo.best_penalty_weights}")

# prediction with the best found parameters
my_bkps = algo.predict(signal=signal_1)
```

----

Utilisation : 
```
from src.cost import KernelWithPartialAnnotationCost, MetricWithPartialAnnotationCost
from src.detect import Pelt_lambda
```

Dans ce qui suit, nous définissons : 

1. Signaux : Liste contenant le jeu de données 
2. bkps : Liste contenant les ruptures du jeu de données

# Apprentissage métrique/noyau

## KernelWithPartialAnnotationCost

```
@dataclass
class TrainingArgs:
    u: float = 0.1
    l: float = 1.0
    gamma: float = 1.0
        
training_args = TrainingArgs()

kernel_init = kernels.DotProduct(sigma_0=0)

cout = noKernelWithPartialAnnotationCost()
cout.pre_fit(
    entrainement,
    labels_train,
    kernel_init
    upper_bound_similarity=training_args.u,
    lower_bound_dissimilarity=training_args.l,
    gamma=training_args.gamma,
)

```

## MetricWithPartialAnnotationCost


```
@dataclass
class TrainingArgs:
    u: float = 0.1
    l: float = 1.0
    gamma: float = 1.0
        
training_args = TrainingArgs()

cout = noKernelWithPartialAnnotationCost()
cout.pre_fit(
    entrainement,
    labels_train,
    upper_bound_similarity=training_args.u,
    lower_bound_dissimilarity=training_args.l,
    gamma=training_args.gamma,
    init=np.diag([1,1,1])
)

```

# Apprentissage pénalité

## Pelt_lambda

```
pelt_lambda=Pelt_lambda(custom_cost=Cout)
pelt_lambda.calcul_penality(signaux,bkps)

```
