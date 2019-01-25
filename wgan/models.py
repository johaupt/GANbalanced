import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, latent_dim, lin_layer_sizes, output_dim, aux_dim=0):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.training_iterations = 0

        # Hidden layers
        first_lin_layer = nn.Linear(latent_dim+aux_dim,
                                    lin_layer_sizes[0])
        self.lin_layers =\
          nn.ModuleList([first_lin_layer] +\
            [nn.Linear(input_, output_) for input_, output_ in zip(lin_layer_sizes, lin_layer_sizes[1:])])

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_dim)

    def forward(self, x, aux_x=None):
        if self.aux_dim != 0:
            x = torch.cat([x,aux_x], dim=1)
        for lin_layer in self.lin_layers:
            x = F.leaky_relu(lin_layer(x), negative_slope=0.2)
            #x = torch.tanh(lin_layer(x))
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        # Return generated data
        return x

    def sample_latent(self, num_samples, class_index=None):
        # Gaussian
        #return torch.randn((num_samples, self.latent_dim))
        # Uniform
        noise = torch.rand((num_samples, self.latent_dim))
        if class_index != None:
            if self.aux_dim ==0:
                warning("self.aux_dim equal 0: Generator does not take auxiliary variables")
                return noise
            aux = torch.zeros([num_samples, self.aux_dim])
            aux[:,class_index] = 1
            return [noise, aux]

        else:
            return noise


class Discriminator(nn.Module):
    def __init__(self, input_size, lin_layer_sizes, aux_input_size=0):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.aux_input_size = aux_input_size

        # Hidden layers
        first_lin_layer = nn.Linear(input_size+aux_input_size,
                                    lin_layer_sizes[0])
        self.lin_layers =\
          nn.ModuleList([first_lin_layer] +\
            [nn.Linear(input_, output_) for input_, output_ in zip(lin_layer_sizes, lin_layer_sizes[1:])])

        self.output_layer = nn.Linear(lin_layer_sizes[-1],1)

    def forward(self, x, aux_x=None):
        #batch_size = x.size()[0]
        if self.aux_input_size != 0:
            x = torch.cat([x,aux_x], dim=1)
        for lin_layer in self.lin_layers:
            x = F.relu(lin_layer(x))
        x = self.output_layer(x)
        return x
