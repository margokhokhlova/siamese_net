# siamese_net with hard-mining for image and vector fused data
The code for my implementation of siamese networks applied to geo data. The model and dataloaders can be found in corresponding files.  
The model is implemented using Keras, the architecture is shown below. There an alternative version without the max pooling as well, in this case maxplooling is replaced by 3 conv and a FC layer.
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/architecture.png)

The architecture is based on classical Siamese netwroks implementations (see Keras siamese Demo or any toher), but modified for my custom data pairs and early fusion. The backbone is ResNet50. Contrastive loss is used in this version.

The dataloaders are all custom, there are two scripts available: training siamese and training siamese hard. The second one pre-calculates the embeddings and creates positive-negative pairs of images using the pre-computed embeddings. The re-computing of hard samples can be performed several times during the training to mine for new hard pairs.

The main files:
model_for_siamese
train_siamese_hard
