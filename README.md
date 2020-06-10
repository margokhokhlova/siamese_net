# A siamese-like model with hard-mining for image and semantic fused data

The code is the reference implementation of a  Siamese architecture applied for cross-temporal matrching of geographical  data used in the publication "Margarita Khokhlova,  Valerie Gouet-Brunet, Nathalie Abadie, and Liming Chen. “Cross-year multi-modal image retrieval  using siamese networks”. To appear at the proceesings of The 27th IEEE International Conference on Image Processing (2020).". 

The architecture proposed is used to learn the descriptors for aerial images of the same geographic zone taken 15 years apart. Both images and semantic labels are used in an early fusion scenario to produce a compact descriptor, which can then be exploited in an image retrieval task.

The model and dataloaders can be found in corresponding files.
The model is implemented using Keras, the architecture is shown below. 

![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/architecture.png)

The architecture is based on classical Siamese netwroks implementations (see Keras siamese Demo or any other), but modified for my custom data pairs and early fusion scenario for multi-modal data (i.e. the network takes two images as an input). The backbone is ResNet50. Binary Cross-Entropy loss is used in this version.

The dataloaders are all custom. The hard mining is performed via pre-calculating embeddings with current network weights and creating positive-negative pairs of images. The re-computing of hard samples can be performed several times during the training to mine for new hard pairs.  In the current implementation hard mining happens each 5 steps.
An example of the input image pairs (positive pairs) is shown below.
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/data.png) 

The main files:
model_for_siamese.py - model definiton
train_siamesee.py -training with hard-mining and an binary cross-entropy or focal loss

We do not provide the final dataset for this work but the unprocessed version of it can be found on the website of ign. The data are called BD TOPO and BD Ortho. https://www.data.gouv.fr/en/datasets/bd-ortho-r-50-cm/. 

Map@5 for unique image correspondence retrieval is used along with the unsuprvised KNN based on computed image descriptors.

The final descriptor dimension can be tuned, I got the best results with the number 128, but 256 also seem to be working fine. The map@5 curves are shown below. 

![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/map@5train.png) 

