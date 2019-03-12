"""Class to perform oversampling with synthetic data from a generative
adversarial network."""

# License: MIT


import numpy as np
from sklearn.utils import check_X_y, check_random_state, safe_indexing

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler

from imblearn.utils import check_target_type
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring

from wgan.data_loader import TabularDataset
from wgan.models_cat import Generator, Critic, make_GANbalancer
from wgan.training import WGAN

from torch.utils.data import DataLoader

from tqdm import tqdm


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class GANbalancer(BaseOverSampler):
    """Class to perform oversampling with synthetic data from a generative
    adversarial network.
    Object to over-sample the minority class(es) by training a GAN and sampling
    from the trained model.
    Parameters
    ----------
    {sampling_strategy}
    {random_state}

     generator_input: int
       The number of random input noise variables (z) to input to the generator

     generator_layers,critic_layers: list of int
       List of nodes in each hidden layer. Can be an empty list to create no
       hidden layer, i.e. a linear model.

     idx_cont:  list of integers
       Column indices of continuous variables.

     categorical: list of tuples (idx_cat, cat_levels, emb_sizes) or None
       List of tuples, one tuple for each categorical variable. Each tuple
       includes the column index of the variable, the number of unique levels within
       the variable (e.g. A, B, A, C -> 3 levels) and the dimensionality of the
       embedding for this variable (e.g. 10 -> 10 latent dimensions).

     auxiliary: True, False
       If False, then any auxiliary variables provided by the dataset are ignored
       and an unconditional Wasserstein GAN is trained.

     batch_size: int
       Batch size for stochastic gradient descent training

     n_iter: int
       Number of training iterations of the generator. The WGAN will be trained
       for as many epochs as are needed to reach the number of training iterations.
       The number of epochs needed depends on the number of observations, batch
       size and critic iterations. When comparing to other research use:
         train_iter = epoch * (no_obs / batch_size*critic_iter)

    critic_iterations: int
       Train critic for n batches before generator is trained on 1 batch.

     learning_rate: 2D tuple of floats
       Learning rate for the (generator, critic)

    Attributes
    ----------

    Notes
    -----
    Supports multi-class resampling by sampling conditional on each class.
    Supports heterogeneous data as object array containing string and numeric
    data.
    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> TODO:
    >>> X_res, y_res = gan.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    def __init__(self, idx_cont, categorical, auxiliary=True,
                 generator_input=10, generator_layers=[10], critic_layers=[10],
                 batch_size=64, n_iter=1000, learning_rate=(5e-5,5e-5),
                 critic_iterations=5,
                 sampling_strategy='auto',
                 random_state=None, verbose=0):
        super().__init__(
            sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.verbose = verbose

        self.idx_cont = idx_cont
        if categorical is not None:
            self.idx_cat, self.cat_levels, self.emb_sizes = map(list,zip(*categorical))
        else:
            self.idx_cat, self.cat_levels, self.emb_sizes = None, None, None
        self.auxiliary = auxiliary

        self.generator_input = generator_input
        self.generator_layers = generator_layers
        self.critic_layers = critic_layers

        self.learning_rate = learning_rate
        self.critic_iterations= critic_iterations

        self.batch_size = batch_size
        self.n_iter = n_iter

        self.generator = None

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=None)
        return X, y, binarize_y

    def _fit(self, X, y=None):
        X_cont=None
        X_cat=None
        if self.idx_cont is not None:
            X_cont = X[:,self.idx_cont]
        if self.idx_cat is not None:
            X_cat = X[:,self.idx_cat]

        dataset = TabularDataset(X = X_cont,
                                    X_cat=X_cat,
                                    y=y,
                                    cat_levels=self.cat_levels
                               )
        self.no_aux = dataset.no_aux
        if self.auxiliary is False:
            self.no_aux = 0

        generator, critic, trainer = make_GANbalancer(
        dataset=dataset, generator_input=self.generator_input,
        generator_layers=self.generator_layers, critic_layers=self.critic_layers,
        emb_sizes=self.emb_sizes, no_aux=self.no_aux, learning_rate = self.learning_rate,
        critic_iterations = self.critic_iterations)

        train_loader = DataLoader(dataset, batch_size = self.batch_size, shuffle=True)

        # Train for generator update iterations instead of epochs, because this is
        # clearer to specify w.r.t to batch size
        if self.verbose>0:  pbar = tqdm(total=self.n_iter)

        while generator.training_iterations < self.n_iter:
            temp_iterations = generator.training_iterations
            trainer._train_epoch(train_loader)

            if self.verbose>0:  pbar.update(generator.training_iterations - temp_iterations)

        if self.verbose>0: pbar.close()

        self.generator = generator
        return self


    def _fit_resample(self, X, y):

        X_resampled = X.copy()
        y_resampled = y.copy()

        random_state = check_random_state(self.random_state)
        #target_stats = Counter(y)

        self._fit(X,y)

        X_new, y_new = self._sample(X,y)

        X_resampled = np.vstack((X_resampled, X_new))
        y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def _sample(self,X,y):

        X_new = np.empty(shape = (0,X.shape[1]))
        y_new = np.array([])

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                 continue

            if self.no_aux == 0:
                class_sample = None
            X_class = self.generator.sample_data(n_samples, class_index=class_sample,
                                               random_state=self.random_state)
            y_class = np.ones(n_samples)*class_sample

            X_new = np.vstack((X_new, X_class))
            y_new= np.hstack((y_new, y_class))

        return X_new, y_new
