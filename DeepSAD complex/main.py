import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer

##############################################################################
# Welcome to DeepSAD! This is essentially a really fancy SVM, in that its original use is for classification between
# binary healthy/unhealthy data. We're going to use the fact it uses a gradient boundary between the two to give us HIs.
# It's semisupervised, which means the training dataset is only partially labelled: 1 for healthy, -1 for unhealthy
#
# The code for DeepSAD itself is from: https://doi.org/10.48550/arXiv.1906.02694
# And you can download the original code from: https://github.com/lukasruff/Deep-SAD-PyTorch
# The method we use to construct HIs is taken from: https://doi.org/10.48550/arXiv.2312.02867
#
# There are three parts to adapting this algorithm to work and give HIs for us:
# 1. Adjust the algorithm to improve the diversity of the embedding (Part 2 of the paper)
#   - Feature fusion is something we'll do elsewhere, so let's ignore that for now
# 2. Figuring out how to apply it to our dataset - we'll have to write an algorithm to give us some labels
# 3. Integration with the rest of the code and testing
#
# Number 1 is already done! I made a short function to produce the embedding (HI) and also edited DeepSAD_trainer.train
# to include improvements in diversity, as mentioned in the paper. Understanding the code and equations took most of the
# time! To help with that I cleaned a few things up, deleted what we don't need and added comments, but there's a long
# way to go still.
#
# Number 2 is the code you can get started on!
#  - I suggest making a new function in the DeepSAD.py file (NOT inside the class definition!) which can take one of our
#    CSVs (of any size, as it might be after feature extraction) and create a separate list of labels
#  - The CSVs are each for one moment in time, so you'll have to include another parameter to let the code know what
#    label to give it. This would ideally by called by a second function which calls the first and tells it which label
#    to use with which data based on when that data is from, but maybe talk to George about how to implement that.
#  - Additionally, the 'X' data can't just be the signals - it either has to be features or some funky transform. It
#    would be great if you could look into this to some extent.
#  - In the 2nd paper they explain how they pick the labels as 1, 0 or -1 based on when the data is from. We could do a
#    similar thing, but we could also consider virtual labels by creating a gradient between 1 and -1, kinda similar to
#    what Moradi did. This is probably something we should ask him about.
#
# Number 3 I have also already had a go at. The DeepSAD.py file is my attempt to recreate what they do in all of these
# files in one much simpler file, just by throwing away unnecessary features. So it will probably be easier for you to
# put your work in that one, but it's all these files which provide the reference for the actual code, so they're all
# here so you can see how it works if you want to. One thing I didn't include was pre-training using autoencoders,
# because I'm not sure how necessary it is or if we can achieve it just by using the AEs the others are making anyway.
#
#
# Overall, it's quite messy here and in the papers but once you get your head around it, everything becomes a lot more
# manageable. That's what will really take the longest time, probably much longer than the coding itself, so don't worry
# if it takes a bit to understand, and feel free to ask any questions. Good luck!
#
# - JJ 10/04/24
##############################################################################

class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    #I think this is to load pre-trained networks so we shouldn't use it
    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, reg: float = 0):
        """Trains the Deep SAD model on the training data. This doesn't actually do the training itself, it just makes a
        trainer object and then calls another function that does.
        I don't know what most of these are, but I added reg which is the hyperparameter lambda for diversification
        """

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net, reg)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder.
        Yeah this needs an autoencoder first for some reason. Not quite sure why but it's mentioned in the papers.
        """

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)


    #New method to return the "embedding". This is the HI!
    #(Improved diversity just changes the model, the embedding is the same)
    def embed(self, X, net):
        """
        Executes the DeepSAD model

        Parameters:
        - X (1D array): Array containing input data, probably features
        - X (BaseNet): Trained neural network

        Returns:
        - y (float): Embedding of X. This is an anomaly score from 0 to 100 (I think).

        Example:
        result = model.embed(features)
        """

        #Some setup functions
        net = net.to(self.device) #Idk what this does but they have it in testing
        X = X.to(self.device)
        net.eval()

        y = torch.norm(net(X) - self.c)   #Magnitude of the vector = anomaly score

        return y