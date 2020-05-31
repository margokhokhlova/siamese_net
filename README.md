# siamese_net with hard-mining for image and vector fused data
The code is the reference implementation of a  Siamese architecture applied for cross-temporal matrching of geographical  data used in the publication "Margarita Khokhlova,  Valerie Gouet-Brunet, Nathalie Abadie, and Liming Chen. “Cross-year multi-modal image retrieval  using siamese networks”. To appear at the proceesings of The 27th IEEE International Conference on Image Processing (2020).". 
The model and dataloaders can be found in corresponding files.
The model is implemented using Keras, the architecture is shown below. 

![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/architecture.png)

The architecture is based on classical Siamese netwroks implementations (see Keras siamese Demo or any other), but modified for my custom data pairs and early fusion scenario for multi-modal data (i.e. the network takes two images as an input). The backbone is ResNet50. Binary Cross-Entropy loss is used in this version.

The dataloaders are all custom. The hard mining is performed via pre-calculating embeddings with current network weights and creating positive-negative pairs of images. The re-computing of hard samples can be performed several times during the training to mine for new hard pairs. In the current implementation hard mining happens each 5 steps.
An example of the input image pairs (positive pairs) is shown below.
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/data.png) 

The main files:
model_for_siamese.py - model definiton
train_siamesee.py -training with hard-mining and an binary cross-entropy or focal loss

Accuracy metric used in map@5 for unique image correspondence retrieval.

The final descriptor dimension can be tuned, I got the best results with the number 128, but 256 also seem to be working fine. The map@5 curves are shown below. 

![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/map@5train.png) 

