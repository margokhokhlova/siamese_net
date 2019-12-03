# siamese_net with hard-mining for image and vector fused data
The code for my implementation of siamese networks applied to geo data. The model and dataloaders can be found in corresponding files.  
The model is implemented using Keras, the architecture is shown below. 
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/architecture.png)

The architecture is based on classical Siamese netwroks implementations (see Keras siamese Demo or any other), but modified for my custom data pairs and early fusion scenario for multi-modal data (i.e. the network takes two images as an input). The backbone is ResNet50. Binary Cross-Entropy loss is used in this version.

The dataloaders are all custom. The hard mining is performed via pre-calculating embeddings with current network weights and creating positive-negative pairs of images. The re-computing of hard samples can be performed several times during the training to mine for new hard pairs.
An example of the input image pairs (positive painrs) is whon below.
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/data.png) 

The main files:
model_for_siamese
train_siamese - training without hard-mining (old version with contrastive loss, not stable)
train_siamese_alternative -training with hard-mining and an binary cross-entropy or focal loss

![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/train_accuracy.png) 
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/train_loss.png)
