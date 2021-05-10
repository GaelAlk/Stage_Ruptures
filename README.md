# Stage Ruptures




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
