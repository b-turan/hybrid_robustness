# Main references
* Hybrid: Ermon/Kuleshov http://www.cs.cornell.edu/~kuleshov/papers/uai2017.pdf 
* OOD via p(y|x): https://arxiv.org/abs/1610.02136
* IWAE: https://arxiv.org/abs/1509.00519
* ECE: https://arxiv.org/abs/1706.04599
* FGSM: https://arxiv.org/abs/1412.6572

Implementation of Deep Hybrid Models (DHMs) consisting of a VAE and a ResNet. 
# Additional Features: 
* Training Datasets: SVHN, CIFAR10
* Adversarial Accuracy in the context of FGSM attacks
* OOD Datasets: SVHN, CIFAR100
* OOD Detection via max p(y|x) as in https://arxiv.org/abs/1610.02136 (using ARUOC)
* OOD Detection via log p(x) (using AUROC)
* Customized VAE with forward hooks for feature level investigation
* Importance Weighted Autoencoders (IWAE) as in https://arxiv.org/abs/1509.00519
* Beta-VAE, see https://openreview.net/forum?id=Sy2fzU9gl
* Checkpoint loading of pretrained ResNet-Classifier with Head 
# Remark
The code was not intended for the public and was therefore not implemented according to specific coding standards.


