# Adversial-Attacks Methods implementation

<p align="justify">This project represent impact of white-box adversarial attacks on Convolutional Neural Networks. Chosen algortims are Fast-Gradient Sign Methods, Jacobian-based Saliency Map Attack, Basic Iterative Method and Iterative Least-Likely Class Method.</p>

<p align="justify">Several models of Convolutional Neural Networks have been developed. First model have been trained on MNIST dataset. Second model have been trained on CIFAR-10 dataset. Accuracy of  first model is 98.56%. Accuracy of second model is 86.91%.</p>

<p align="justify">Classes FGSM, JSMA, BIM, ILLCM implement interface AdversarialAttack and represent aforementioned adversarial algoritms.</p>
<p align="justify">Jupyter notebook script adversarial_experiments.ipynb represent experiments with adversarial attacks.</p>

<p align="center"><img src="https://user-images.githubusercontent.com/67442675/121735659-b6a58b00-caf6-11eb-9b32-3c63733f6c86.jpg">Experiment example</p>

<p align="justify">Files CNN_adv_cifar.py and CNN_adv_mnist.py represents adversarial attacks defense system.</p>

### Neural Networks

* CNN_model_cifar.json and CNN_model_mnist.json are files with model structure,
* CNN_model_cifar.h5 and CNN_model_mnist.h5 are files with network experience(learned weights),
* CNN_model_cifar_adversarial.json and CNN_model_mnist_adversarial.json are files with protected models structure,
* CNN_model_cifar_adversarial.h5 and CNN_model_mnist_adversarial.h5 are files with re-trained models experience.

