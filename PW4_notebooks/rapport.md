# Rapport LABO4 Learning with Artificial Neural Networks
## ARN
## Auteur: 
## Bastien Pillonel
## Loïc Brasey

\pagebreak

## Question 1:

_What is the learning algorithm being used to optimize the weights of the neural
networks? What are the parameters (arguments) being used by that algorithm? What
cost function is being used ? please, give the equation(s)_

L'algorithm utilisé est le RMSprop de la librairie Keras.

Les paramètres utilisés sont :

```{python}
tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=100,
    jit_compile=True,
    name="RMSprop",
    **kwargs
)
```

L'equation utilisé est :

$v_{dw} = \beta * v_{dw} + (1 - \beta) * dw^2$   
$v_{db} = \beta * v_{dw} + (1 - \beta) * db^2$    
$W = W - a * \frac {dw} {\sqrt {v_{dw}} + \epsilon}$      
$b = b - a * \frac {db} {\sqrt {v_{db}} + \epsilon}$    

Selon article de site web [towardsdatascience](https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b)

Et la loss function utilisé est la [categorical_crossentropy](https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class).

## Question 2

_Model complexity: for each experiment (shallow network learning from raw data, shallow
network learning from features, CNN, and Fashion MNIST), select a neural network
topology and describe the inputs, indicate how many are they, and how many outputs.
Compute the number of weights of each model (e.g., how many weights between the
input and the hidden layer, how many weights between each pair of layers, biases, etc..)
and explain how do you get to the total number of weights._

### Shallow network learning from raw data
Nbr d'entrés : 784\
Nbr de sorties : 10 classes\
Nbr de couches cachée : 1\
Nbr de neurones cachés : 300\
Nbr de poids dans couche caché : 784 * 300 + 300 = 235 500
Nbr de poids à la sortie : 300 * 10 + 10 = 3010
Nbr de poids total : 235 500 + 3010 = 238 510 

### Shallow network learning from features
Nbr d'entrés : 392\
Nbr de sorties : 10 classes\
Nbr de couches cachée : 1\
Nbr de neurones cachés : 200\
Nbr de poids dans couche caché : 392 * 200 + 200 = 78'600
Nbr de poids à la sortie : 200 * 10 + 10 = 2'010
Nbr de poids total :78'600 + 2'010 = 80'610
### CNN
### Fashion MNIST

## Question 3

_Do the deep neural networks have much more “capacity” (i.e., do they have more
weights?) than the shallow ones? explain with one example_

## Question 4

_Test every notebook for at least three different meaningful cases (e.g., for the MLP
exploiting raw data, test different models varying the number of hidden neurons, for the
feature-based model, test pix_p_cell 4 and 7, and number of orientations or number of
hidden neurons, for the CNN, try different number of neurons in the feed-forward part)
describe the model and present the performance of the system (e.g., plot of the
evolution of the error, final evaluation scores and confusion matrices). Comment the
differences in results. Are there particular digits that are frequently confused?_

### Shallow network learning from features

Pour ce notebook nous avons principalement essayé de faire varier le nombre de neurones dans la couches cachée.

**Résultat avec 100 neurones:**

\center
![](./Capture/Graph_loss_raw_100.png){ width=60% }
\center

\center
![](./Capture/Conf_matrix_raw_100.png){ width=60% }
\center

**Résultat avec 200 neurones:**

\center
![](./Capture/Screenshot%20from%202023-05-05%2011-20-32.png){ width=60% }
\center

\center
![](./Capture/Screenshot%20from%202023-05-05%2011-20-46.png){ width=60% }
\center

**Résultat avec 400 neurones:**

\center
![](./Capture/Screenshot%20from%202023-05-05%2011-27-21.png){ width=60% }
\center

\center
![](./Capture/Screenshot%20from%202023-05-05%2011-27-36.png){ width=60% }
\center

**Résultat avec 800 neurones:**

\center
![](./Capture/Screenshot%20from%202023-05-05%2011-33-35.png){ width=60% }
\center

\center
![](./Capture/Screenshot%20from%202023-05-05%2011-33-46.png){ width=60% }
\center

\raggedright
**Interprétation des résultats:**

A partir de 400 neurones, on commence à observer un overfitting du modèle. Encore plus présent pour une couche cachée possèdant 800 neurones (l'erreur de la courbe de test commence à remonter après plusieurs epochs).

Pour ce cas-ci 200 neurones semble être un bon nombre de neurones.

## Question 5

_Train a CNN to solve the MNIST Fashion problem, present your evolution of the errors
during training and perform a test. Present a confusion matrix, accuracy, F-score and
discuss your results. Are there particular fashion categories that are frequently
confused?_