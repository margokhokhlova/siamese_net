# siamese_net
The code for my implementation of siamese networks applied to geo data. The model and dataloaders can be found in corresponding files.  
The model is implemented using Keras, the architecture is shown below. 
![alt text](https://github.com/margokhokhlova/siamese_net/blob/master/architecture.png)

The architecture is based on classical Siamese netwroks implementations (see Keras siamese Demo or any toher), but modified for my custom data pairs and early fusion. The backbone is ResNet50. Contrastive loss is used in this version.
