import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import warnings

from wgan.training import WGAN


def make_GANbalancer(dataset, generator_input, generator_layers, critic_layers,
emb_sizes, no_aux,learning_rate):
    """
    Make a generator and critic to fit the given dataset

    Input
    -----
    dataset: pyTorch Dataset derivative TabularDataset

    generator_layers, critic_layers: list of int
      Number of nodes in each of the hidden layers. Input and Output layers of
      correct size for the data are added automatically

    emb_sizes: list of int
      Embedding dimensionality for each of the categorical variables in the order
      they appear in the dataset
    """
    generator = Generator(latent_dim=generator_input, lin_layer_sizes=generator_layers,
                      output_dim=dataset.no_cont, cat_output_dim=dataset.cat_levels,
                      aux_dim=no_aux)

    cat_inputs = None
    if dataset.cat_levels is not None:
        cat_inputs = list(zip(dataset.cat_levels, emb_sizes))

    critic = Critic(lin_layer_sizes=critic_layers,
                    input_size=dataset.no_cont,cat_input_sizes=cat_inputs,
                    aux_input_size=no_aux)

    betas = (.9, .99)
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate[0], betas=betas)
    C_optimizer = optim.Adam(critic.parameters(), lr=learning_rate[1], betas=betas)

    trainer = WGAN(generator=generator, critic=critic,
                G_optimizer=G_optimizer, C_optimizer=C_optimizer,
                gp_weight=10, critic_iterations=5,
                verbose=0, print_every=1,
                use_cuda=torch.cuda.is_available())

    return generator, critic, trainer

class Generator(nn.Module):
    def __init__(self, latent_dim, lin_layer_sizes, output_dim, cat_output_dim=0, aux_dim=0):
        """
        cat_output_dim (list of integers):
            List of number of levels for each categorical variable in the same
            order as in the real data.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim
        self.cat_output_dim = cat_output_dim
        self.training_iterations = 0

        # Hidden layers
        first_lin_layer = nn.Linear(latent_dim+aux_dim,
                                    lin_layer_sizes[0])
        self.lin_layers =\
          nn.ModuleList([first_lin_layer] +\
            [nn.Linear(input_, output_) for input_, output_ in zip(lin_layer_sizes, lin_layer_sizes[1:])])

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_dim)
        if cat_output_dim !=0 and cat_output_dim is not None:
            self.cat_output_layer = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(lin_layer_sizes[-1],output_),
                nn.Softmax(dim=1)
                ) for output_ in cat_output_dim]
            )

    def forward(self, x, aux_x=None):
        if self.aux_dim != 0:
            x = torch.cat([x,aux_x], dim=1)
        for lin_layer in self.lin_layers:
            x = F.leaky_relu(lin_layer(x), negative_slope=0.1)
            #x = torch.tanh(lin_layer(x))

        # Continuous
        x_cont = self.output_layer(x)
        x_out = torch.sigmoid(x_cont)

        # Categorical
        if self.cat_output_dim!=0 and self.cat_output_dim is not None:
            x_cat = [layer(x) for layer in self.cat_output_layer]
            x_out = torch.cat([x_out, *x_cat], dim=1)

        # Return generated data
        return x_out

    def sample_data(self, num_samples, class_index=None, random_state=None):

        if random_state is not None:
            torch.manual_seed(random_state)
            if cuda:
                torch.cuda.manual_seed(random_state)
        noise = torch.rand((num_samples, self.latent_dim))
        aux = None

        if class_index != None:
            if self.aux_dim !=0:
                aux = torch.zeros([num_samples, self.aux_dim])
                aux[:,class_index] = 1
            else:
                warnings.warn("self.aux_dim equal 0: Generator does not take auxiliary variables")

        x = self(noise, aux_x=aux)

        # Sample from a categorical distribution
        if self.cat_output_dim!=0 and self.cat_output_dim is not None:
            i = self.output_dim
            x_ordinal = []
            for layer_id, levels in enumerate(self.cat_output_dim):
                j = i+levels
                x_ordinal.append( torch.multinomial(x[:,i:j],1).float()  )
                i = j
            x = torch.cat([x[:,:self.output_dim], *x_ordinal], dim=1)

        return x.data.numpy()

    def sample_latent(self, num_samples, class_index=None):
        # Gaussian
        #return torch.randn((num_samples, self.latent_dim))
        # Uniform
        noise = torch.rand((num_samples, self.latent_dim))
        if class_index != None:
            if self.aux_dim ==0:
                warnings.warn("self.aux_dim equal 0: Generator does not take auxiliary variables")
                return noise
            aux = torch.zeros([num_samples, self.aux_dim])
            aux[:,class_index] = 1
            return [noise, aux]

        else:
            return noise


class Critic(nn.Module):
    def __init__(self, input_size, lin_layer_sizes, cat_input_sizes=0, aux_input_size=0):
        """
        input_size (integer):
            Number of continous variables in the input data

        cat_input_sizes (list of tuples):
            One tuple for each variable specifying (number of levels, embedding
            size)
        """
        super().__init__()

        self.input_size = input_size
        self.cat_input_sizes = cat_input_sizes

        if cat_input_sizes != 0 and cat_input_sizes is not None:
            self.embedding_size = sum([y for x,y in cat_input_sizes])
            # Embedding layers
            self.emb_layers = nn.ModuleList([nn.Linear(x, y, bias=False)
                                     for x, y in cat_input_sizes])
        else:
            self.embedding_size=0

        self.aux_input_size = aux_input_size

        # Hidden layers
        first_lin_layer = nn.Linear(input_size+self.embedding_size+aux_input_size,
                                    lin_layer_sizes[0])
        self.lin_layers =\
          nn.ModuleList([first_lin_layer] +\
            [nn.Linear(input_, output_) for input_, output_ in zip(lin_layer_sizes, lin_layer_sizes[1:])])

        self.output_layer = nn.Linear(lin_layer_sizes[-1],1)

    def forward(self, x, aux_x=None):
        #batch_size = x.size()[0]

        if self.embedding_size != 0:
            i = self.input_size
            x_emb = []
            for layer_id, (levels, _) in enumerate(self.cat_input_sizes):
                j = i+levels
                x_emb.append(self.emb_layers[layer_id](x[:,i:j]))
                i = j
            x = torch.cat([x[:,:self.input_size], *x_emb], dim=1)

        if self.aux_input_size != 0:
            x = torch.cat([x,aux_x], dim=1)

        for lin_layer in self.lin_layers:
            x = F.relu(lin_layer(x))
        x = self.output_layer(x)
        return x
