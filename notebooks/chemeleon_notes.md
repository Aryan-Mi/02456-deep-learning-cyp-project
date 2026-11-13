# CheMeleon model notes (for the report)

CheMeleon is a O(10M) parameters foundation model build upon direct-message-passing architectre (D-MPNN) from Chemprop.

* Training : Trained on 1M molecules from PubChem to predict their Mordred descriptors (numerical values characterizing the molecule)

* Data flow : SMILES (strings) -> Molecule graph (RDKit, Mol) -> Chemprop featurizer -> Molecule graph with feaure matrices (MolGraph) -> CheMeleon encoder (atom embeddings) -> Mean aggregation (Molecule embeddings : 2048) -> readout head  

* Fine tuning : was trained to predict the descriptors, then the FNN head was discarded and only the encoder weights are saved. Then for the finetuning we import the weights and add our custom heads. 


### Why we chose Chemeleon ?

1) It was reported to perform well on low-range data, which is true in our case 

2) It works on 2D structure instead of 3D - will be a good comparison agains 3D Atom-Jepa

3) Is possible to finetune for our tasks 

4) We will try it on multilabel tasks, which performance was not tested in the article. 

### Limitations of this model:

1) Activity cliffs - mentioned in the article; poor distinction between molecules that have a similar structure but different activities. This is however on a limitation of only this model but overall foundation models problem.

2) Featurizer selection - we have to use the same Featurizer from Chemprop

3) A lot of model parameters - possible quick overfitting 





