import torch
from torch import nn, optim
# Variable provides a wrapper around tensors to allow automatic differentiation, etc.
from torch.autograd.variable import Variable
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np

def make_noise(size, dim, ratio_binomial=False):
    """
    Generates a vector with length 100 of Gaussian noise with (batch_size, 100)

    Parameters
    ----------
    size: Integer
      Number of observations in batch

    dim: Integer
      Length of the noise vector

    ratio_binomial: None or float [0;1]
      Ratio of noise variables drawn from a binomial distribution instead of a
      standard normal distribution
    """
    if ratio_binomial is False:
        n = Variable(
            torch.randn(size, dim) # random values from standard normal
        )
    else:
        n = Variable(
            torch.cat([
                torch.randn(int(size*ratio_binomial), dim),
                torch.from_numpy(np.random.binomial(n=1, p=0.5, size=[int(size*ratio_binomial),dim])).float()
            ])
        )

    return n


class GeneratorNet(torch.nn.Module):
    """
    Generator network to create artificial data from random noise input.

    Parameters
    ----------
    noise_dim: integer
      Number of random values fed as input to the generator

    lin_layer_sizes: List of integers.
      The size of each linear layer. The length will be equal
      to the total number
      of linear layers in the network.

    lin_layer_dropouts: List of floats or None
      The dropouts to be used after each linear layer. None for no dropout.

    no_of_cont: Integer
      The number of continuous features in the data.

    emb_dims: List of two element tuples
      This list will contain a two element tuple for each
      categorical feature. The first element of a tuple will
      denote the number of unique values of the categorical
      feature. The second element will denote the embedding
      dimension to be used for that feature.

    Output
    ------
    Returns a list of two numpy arrays [cont_data, cat_data] that can be used
    directly as input to a critic. Categorical variables are expressed as softmax
    output, i.e. a vector of probabilities that sum up to 1.
    To generate a pre-processed dataframe use function sample() instead.
    """
    def __init__(self, noise_dim, lin_layer_sizes, lin_layer_dropouts,
                    no_of_cont, emb_dims):
        super().__init__()

        self.noise_dim = noise_dim
        self.no_of_cont = no_of_cont
        #self.n_output_binary = n_output_binary
        no_of_cat = sum([x for x, y in emb_dims])
        self.no_of_cat = no_of_cat

        # Hidden layers
        first_lin_layer = nn.Linear(noise_dim,
                                    lin_layer_sizes[0])

        self.lin_layers =\
          nn.ModuleList([first_lin_layer] +\
            [nn.Linear(input_, output_) for input_, output_ in zip(lin_layer_sizes, lin_layer_sizes[1:])])

        if lin_layer_dropouts:
            self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                          for size in lin_layer_dropouts])
        else:
            self.droput_layers = None

        # Output layers

        # if self.n_output_binary is not None:
        #     self.binary = nn.ModuleList()
        #     for x in range(n_output_binary):
        #         self.binary.append(
        #             nn.Sequential(
        #                 nn.Linear(hidden_layers[-1], 1),
        #                 nn.Sigmoid()
        #             )
        #         )

        if self.no_of_cont != 0:
            self.continuous = nn.Sequential(
                nn.Linear(lin_layer_sizes[-1], no_of_cont)#,
                #nn.Sigmoid()
            )

        if self.no_of_cat != 0:
            self.categorical = nn.ModuleList()
            for x,_ in emb_dims:
                self.categorical.append(
                    nn.Sequential(
                        nn.Linear(lin_layer_sizes[-1], x),
                        nn.Softmax(dim=1)
                    )
                )

    def forward(self, x):

        # hidden layers (with/out dropout)
        if self.droput_layers:

            for lin_layer, dropout_layer in\
                zip(self.lin_layers, self.droput_layers):

                x = F.relu(lin_layer(x))
                #x = bn_layer(x)
                x = dropout_layer(x)
        else:
            for lin_layer in self.lin_layers:
                x = F.relu(lin_layer(x))

        #for i_hidden in range(len(self.hidden)):
        #    x = self.hidden[i_hidden](x)

        # out_continuous = None
        # out_binary = None
        # out_categorical = None

        # if self.n_output_binary is not None:
        #     out_binary = [self.binary[var](x) for var in
        #                range(self.n_output_binary)]
        #     if len(out_binary) > 1:
        #         out_binary = torch.cat(*out_binary, dim=1)
        #     else:
        #         out_binary = out_binary[0]

        if self.no_of_cat != 0:
            out_categorical = [layer(x) for layer in self.categorical]
                               # [self.categorical[var](x)
                               # for var in
                               # range(len(self.n_output_categorical))]
            if len(out_categorical) > 1:
                cat_data = torch.cat(out_categorical, dim=1)
            else:
                cat_data = out_categorical[0]
        else:
            cat_data = None

        if self.no_of_cont != 0:
            cont_data = self.continuous(x)
        else:
            cont_data = None

        # output = [x for x in [out_continuous, out_binary, out_categorical] if x is not None]
        # if len(output)>1:
        #     output = torch.cat(output, dim=1)
        # else:
        #     output = output[0]

        return [cont_data, cat_data]

    def sample(self, x):
        """
        Sample artifical data from the generator in the shape of the original
        data. Categorical variables are sampled from a multinomial distribution
        each with probabilities equal to the softmax output.
        """

        self.eval()
        # hidden layers (with/out dropout)
        if self.droput_layers:

            for lin_layer, dropout_layer in\
                zip(self.lin_layers, self.droput_layers):

                x = F.relu(lin_layer(x))
                #x = bn_layer(x)
                x = dropout_layer(x)
        else:
            for lin_layer in self.lin_layers:
                x = F.relu(lin_layer(x))

        # Categorical variables
        # Sample according to predicted probabilities
        if self.no_of_cat != 0:
            out_categorical = [layer(x) for layer in self.categorical]
            out_categorical = [torch.eye(output.shape[1])[torch.multinomial(
                   output,1).squeeze()]
                for output in out_categorical]
            # Deal with one or more than one categorical variable
            if len(out_categorical) > 1:
                out_categorical = torch.cat(out_categorical, dim=1)
            else:
                out_categorical = out_categorical[0]
        else:
            out_categorical = None

        # Continuous variables
        if self.no_of_cont != 0:
            out_continuous = self.continuous(x)
        else:
            out_continuous = None

        output = [x for x in [out_continuous, out_categorical] if x is not None]
        if len(output)>1:
            output = torch.cat(output, dim=1)
        else:
            output = output[0]

        return output

        # if self.n_output_binary is not None:
        #     out_binary = [torch.bernoulli(self.binary[var](x)).float() for var in
        #                range(self.n_output_binary)]
        #     if len(out_binary) > 1:
        #         out_binary = torch.cat(*out_binary, dim=1)
        #     else:
        #         out_binary = out_binary[0]

        # if self.no_of_cat != 0:
        #     out_categorical = [torch.eye(self.n_output_categorical[var])[torch.multinomial(self.categorical[var](x),1).squeeze()]
        #                         #[torch.eye(self.n_output_categorical[var])[torch.argmax(self.categorical[var](x), dim=1)]
        #                        for var in
        #                        range(len(self.n_output_categorical))]
        #     if len(out_categorical) > 1:
        #         out_categorical = torch.cat(*out_categorical, dim=1)
        #     else:
        #         out_categorical = out_categorical[0]
        #
        # if self.no_of_cont != 0:
        #     out_continuous = self.continuous(x)
        #
        # output = [x for x in [out_continuous, out_binary, out_categorical] if x is not None]
        # if len(output)>1:
        #     output = torch.cat(output, dim=1)
        # else:
        #     output = output[0]
        #
        # return output
        #
        # out_binary = [(self.binary[var](x)>0.5).float() for var in
        #            range(self.n_output_binary)]
        # out_categorical = [torch.eye(self.n_output_categorical[var])[torch.multinomial(self.categorical[var](x),1).squeeze()]
        #                     #[torch.eye(self.n_output_categorical[var])[torch.argmax(self.categorical[var](x), dim=1)]
        #                    for var in
        #                    range(len(self.n_output_categorical))]
        #
        # out_continuous = self.continuous(x)
        # return torch.cat((out_continuous,*out_binary, *out_categorical), dim=1)


class CriticNet(torch.nn.Module):
    def __init__(self, no_of_cont, emb_dims, lin_layer_sizes,
                    emb_dropout, lin_layer_dropouts):

        """
        A feed-forward neural network with one-dimensional continuous output

        Parameters
        ----------
        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        no_of_cont: Integer
          The number of continuous features in the data.

        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        #self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
        #                                 for x, y in emb_dims])
        self.emb_layers = nn.ModuleList([nn.Linear(x, y, bias=False)
                                         for x, y in emb_dims])


        # Total input dimension (cont variables + embedding dimensions)
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont
        self.emb_dims = emb_dims

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers =\
         nn.ModuleList([first_lin_layer] +\
              [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
               for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],1)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # No batch norm for Wasserstein or Cramer GAN
        # Batch Norm Layers
        #self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        #self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
        #                                for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                      for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            #x = [emb_layer(cat_data[:, i])\
            #       for i,emb_layer in enumerate(self.emb_layers)]
            col_idx = 0
            x = []
            for layer_no, (levels, emb_dim) in enumerate(self.emb_dims):
                x.append(self.emb_layers[layer_no](cat_data[:, col_idx:col_idx+levels]))
                col_idx += levels
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)


        if self.no_of_cont != 0:
            #normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, cont_data], 1)
            else:
                x = cont_data

        for lin_layer, dropout_layer in\
            zip(self.lin_layers, self.droput_layers):

            x = F.leaky_relu(lin_layer(x), negative_slope=0.2)
            #x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x


def interpolate_data(real_data, fake_data):
    # Check if data is a single array or list of (cont_data, cat_data)
    if isinstance(real_data, (list, tuple)):
        eps = torch.rand(real_data[0].size(0), 1) # A random unif number for each obs in the batch
        output = []

        for real_vars, fake_vars in zip(real_data, fake_data):
            eps_vars = eps.expand(real_vars.size()) # Can only multiply tensors with tensors, so expand to same dimensions
            interpolated_data = eps*real_vars + (1-eps)*fake_vars
            interpolated_data = Variable(interpolated_data, requires_grad=True) # Transform into Variable again
            output.append(interpolated_data)
        return output

    else:
        eps = torch.rand(real_data.size(0), 1) # A random unif number for each obs in the batch
        eps = eps.expand(real_data.size()) # Can only multiply tensors with tensors, so expand to same dimensions
        interpolated_data = eps*real_data + (1-eps)*fake_data
        interpolated_data = Variable(interpolated_data, requires_grad=True) # Transform into Variable again
        return interpolated_data

def critic_loss(output_real, output_fake):
    """
    Wasserstein distance to minimize as loss for the critic
    -(E[D(x_real)] - E[D(x_fake)])
    """
    return -( torch.mean(output_real) - torch.mean(output_fake) )

def merge_data_types(data):
    """
    Funtion to concatenate output of generator split by variable types into one

    Parameters
    ----------
    data: List of arrays
    Typically output of generator of the form [cont_data, cat_data], where each
    element can be None if the data type does not exist in the dataset
    """
    output = [x for x in data if x is not None]
    if len(output)>1:
        output = torch.cat(output, dim=1)
    else:
        output = output[0]
    return output

class WGAN():
    """
    Wrapper around a generator and critic Wasserstein GAN.
    """
    def __init__(self, generator, critic):
        self.generator = generator
        self.critic = critic
        self.n_epoch = 0

    def train_WGAN(self, data_loader, critic_optimizer, generator_optimizer,
    num_epochs, critic_rounds=5, gradient_penalty_coefficient=10,val_data=None):
        critic_performance = []
        generator_performance = []

        for epoch in range(num_epochs):
            # enumerate() outputs index, value for an indexable object
            # output the index of the batch and the output of data_loader
            # data_loader() outputs a batch of images and their label (which we don't need in this case)
            for n_batch, (label, real_cont, real_cat) in enumerate(data_loader):
                N = real_cont.size(0) # Get the number of images from tensor
                if val_data is not None:
                    N_val = val_data[0].size(0)

                ## Train discriminator
                # Collect real data
                real_cont = Variable(real_cont.float())
                real_cat = Variable(real_cat.float())
                real_data = [real_cont, real_cat]

                temp_performance = []
                for k in range(critic_rounds):
                    # Create fake data
                    # generator() creates a graph on the fly, which we drop after collecting the fake data
                    fake_data = [x.detach() for x in self.generator(make_noise(N, self.generator.noise_dim))]
                    # Start critic training
                    disc_error, disc_pred_real, disc_pred_fake = self.train_critic(real_data = real_data, fake_data = fake_data,
                                                                              optimizer = critic_optimizer,
                                                                              gradient_penalty_coefficient=gradient_penalty_coefficient)
                    #temp_performance.append(disc_error.detach().cpu().numpy())
                #critic_performance.append(-np.mean(temp_performance))
                critic_performance.append(-disc_error.detach().cpu().numpy())

                ## Train generator
                # This time we keep the graph, because we backprop on it in the training function
                gen_error = self.train_generator(optimizer = generator_optimizer, N = N)
                generator_performance.append(gen_error.detach().cpu().numpy())

                if n_batch % 50 ==0:
                    fake_data = self.generator(make_noise(N, self.generator.noise_dim))
                    if val_data is None:
                        print("{train:.6f} | ".format(
                            train=(self.critic(cont_data = real_data[0], cat_data = real_data[1]).mean() -\
                             self.critic(cont_data = fake_data[0], cat_data = fake_data[1]).mean()).detach().cpu().numpy()
                        ))
                    else:
                        print("{train:.6f} | {val:.6f}".format(
                            train=(self.critic(cont_data = real_data[0], cat_data = real_data[1]).mean() -\
                             self.critic(cont_data = fake_data[0], cat_data = fake_data[1]).mean()).detach().cpu().numpy(),

                            val  =(self.critic(cont_data = val_data[0], cat_data = val_data[1]).mean()  -\
                             self.critic(self.generator(make_noise(N_val, self.generator.noise_dim))).mean()).detach().cpu().numpy()
                        ))

                    #print(pd.DataFrame(generator(validation_noise).detach().numpy()).mean())
            self.n_epoch += 1
        return critic_performance, generator_performance

    def train_critic(self, optimizer, real_data, fake_data, gradient_penalty_coefficient=10):
        N = real_data[0].size(0) # Get number of rows from torch tensor

        #for param in self.critic.parameters():
        #    param.requires_grad = True

        # Note: Calling backward() multiple times will acumulate the gradients
        # until they are reset with zero_grad()
        # E[D(x_real)]
        output_real = self.critic(cont_data = real_data[0], cat_data = real_data[1])

        # E[D(x_fake)]
        output_fake = self.critic(cont_data = fake_data[0], cat_data = fake_data[1])

        # - (E[D(x_real) - E[D(x_fake)]]
        raw_loss = torch.mean(output_fake) - torch.mean(output_real)

        # Gradient penalty
        # To calculate the gradient penalty, the interpolated data requires
        # a joint data array
        gradient_penalty = self.calc_gradient_penalty(real_data,
                                                 fake_data)

        # Calculate overall loss
        # Minimize the raw loss pushed upwards by penalty (always positive)
        optimizer.zero_grad() # reset gradient
        loss = raw_loss + gradient_penalty_coefficient * gradient_penalty.mean()

        # Weight update
        loss.backward()
        optimizer.step()

        # Return error and predictions for monitoring
        return raw_loss, output_real, output_fake

    def train_generator(self, optimizer, N):
        #for param in self.critic.parameters():
        #    param.requires_grad = False
        optimizer.zero_grad() # reset gradient

        # Create fake data
        fake_data = self.generator(make_noise(N, dim = self.generator.noise_dim))

        # Get discriminator prediction output
        critic_prediction = self.critic(cont_data = fake_data[0],
                                                cat_data = fake_data[1])

        # See explanation above. Intuitively, we create loss if the
        # discriminator predicts our pseudo-ones as zeros.
        loss_generator = -critic_prediction.mean()
        loss_generator.backward()

        # Weight update
        optimizer.step()

        # Return error and predictions for monitoring
        return loss_generator

    def calc_gradient_penalty(self, real_data, fake_data):
        interpolated_data = interpolate_data(real_data, fake_data)
        batch_size = real_data[0].size(0)

        critic_output = self.critic(cont_data = interpolated_data[0], cat_data = interpolated_data[1])
        gradients = autograd.grad(inputs=[interpolated_data[0], interpolated_data[1]], outputs=critic_output,
                                 grad_outputs=torch.ones(critic_output.size()),
                                  create_graph=True, retain_graph=True)[0] #, only_inputs=True

        gradients = gradients.view(batch_size, -1)
        gradient_penalty=((gradients.norm(2, dim=1)-1) ** 2)
        return gradient_penalty
