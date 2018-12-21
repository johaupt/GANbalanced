import torch
from torch import nn, optim
# Variable provides a wrapper around tensors to allow automatic differentiation, etc.
from torch.autograd.variable import Variable
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader

import numpy as np

def make_noise(size, dim, binary=True):
    """
    Generates a vector with length 100 of Gaussian noise with (batch_size, 100)
    """
    if binary is False:
        n = Variable(
            torch.randn(size, dim) # random values from standard normal
        )
    else:
        n = Variable(
            torch.cat([
                torch.randn(3*size//4, dim),
                torch.from_numpy(np.random.binomial(n=1, p=0.5, size=[size//4,dim])).float()
            ])
        )

    return n


class GeneratorNet(torch.nn.Module):
    """
    A three-layer generative neural network
    Assumes that the input data is sorted continuous, then categorical
    output_continuous: Number of continuous variables
    output_continuous: Number of binary variables
    output_categorical: List of number of columns each categorical variable
    """
    def __init__(self, hidden_layers, n_output_continuous=None, n_output_binary=None,
    n_output_categorical=None, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_output_continuous = n_output_continuous
        self.n_output_binary = n_output_binary
        self.n_output_categorical = n_output_categorical

        layer_dimensions = [noise_dim, *hidden_layers]
        self.hidden = nn.ModuleList()

        # Hidden layers
        for input_, output_ in zip(layer_dimensions, layer_dimensions[1:]):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(input_, output_),
                    nn.LeakyReLU(0.2)
                    # TODO: Why no dropout in generator?
                )
            )

        # Output layers
        if self.n_output_binary is not None:
            self.binary = nn.ModuleList()
            for x in range(n_output_binary):
                self.binary.append(
                    nn.Sequential(
                        nn.Linear(hidden_layers[-1], 1),
                        nn.Sigmoid()
                    )
                )

        if self.n_output_categorical is not None:
            self.categorical = nn.ModuleList()
            for x in n_output_categorical:
                self.categorical.append(
                    nn.Sequential(
                        nn.Linear(hidden_layers[-1], x),
                        nn.Softmax(dim=1)
                    )
                )

        if self.n_output_continuous is not None:
            self.continuous = nn.Sequential(
                nn.Linear(hidden_layers[-1], n_output_continuous),
            )

    def forward(self, x):
        for i_hidden in range(len(self.hidden)):
            x = self.hidden[i_hidden](x)
        # x = self.hidden0(x)
        # x = self.hidden1(x)
        # x = self.hidden2(x)

        out_continuous = None
        out_binary = None
        out_categorical = None

        if self.n_output_binary is not None:
            out_binary = [self.binary[var](x) for var in
                       range(self.n_output_binary)]
            if len(out_binary) > 1:
                out_binary = torch.cat(*out_binary, dim=1)
            else:
                out_binary = out_binary[0]

        if self.n_output_categorical is not None:
            out_categorical = [self.categorical[var](x)
                               for var in
                               range(len(self.n_output_categorical))]
            if len(out_categorical) > 1:
                out_categorical = torch.cat(*out_categorical, dim=1)
            else:
                out_categorical = out_categorical[0]

        if self.n_output_continuous is not None:
            out_continuous = self.continuous(x)

        output = [x for x in [out_continuous, out_binary, out_categorical] if x is not None]
        if len(output)>1:
            output = torch.cat(output, dim=1)
        else:
            output = output[0]

        return output

    def sample(self, x):
        for i_hidden in range(len(self.hidden)):
            x = self.hidden[i_hidden](x)

        out_continuous = None
        out_binary = None
        out_categorical = None

        if self.n_output_binary is not None:
            out_binary = [torch.bernoulli(self.binary[var](x)).float() for var in
                       range(self.n_output_binary)]
            if len(out_binary) > 1:
                out_binary = torch.cat(*out_binary, dim=1)
            else:
                out_binary = out_binary[0]

        if self.n_output_categorical is not None:
            out_categorical = [torch.eye(self.n_output_categorical[var])[torch.multinomial(self.categorical[var](x),1).squeeze()]
                                #[torch.eye(self.n_output_categorical[var])[torch.argmax(self.categorical[var](x), dim=1)]
                               for var in
                               range(len(self.n_output_categorical))]
            if len(out_categorical) > 1:
                out_categorical = torch.cat(*out_categorical, dim=1)
            else:
                out_categorical = out_categorical[0]

        if self.n_output_continuous is not None:
            out_continuous = self.continuous(x)

        output = [x for x in [out_continuous, out_binary, out_categorical] if x is not None]
        if len(output)>1:
            output = torch.cat(output, dim=1)
        else:
            output = output[0]

        return output

        out_binary = [(self.binary[var](x)>0.5).float() for var in
                   range(self.n_output_binary)]
        out_categorical = [torch.eye(self.n_output_categorical[var])[torch.multinomial(self.categorical[var](x),1).squeeze()]
                            #[torch.eye(self.n_output_categorical[var])[torch.argmax(self.categorical[var](x), dim=1)]
                           for var in
                           range(len(self.n_output_categorical))]

        out_continuous = self.continuous(x)
        return torch.cat((out_continuous,*out_binary, *out_categorical), dim=1)



class CriticNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, hidden_layers, input_dim, dropout=0.2):
        super().__init__() # get the __init__() from the parent module
        self.input_dim = input_dim

        layer_dimensions = [input_dim, *hidden_layers]
        self.hidden = nn.ModuleList()

        for input_, output_ in zip(layer_dimensions, layer_dimensions[1:]):
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(input_, output_),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout)
                )
            )

        # self.hidden0 = nn.Sequential(
        #     nn.Linear(input_dim, 256), # Linear transformation part input*W+b
        #     nn.LeakyReLU(0.2), # leaky relu is more robust for GANs than ReLU
        #     nn.Dropout(dropout)
        #     )
        #
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(256, 256), # Linear transformation part input*W+b
        #     nn.LeakyReLU(0.2), # leaky relu is more robust for GANs than ReLU
        #     nn.Dropout(dropout)
        #     )
        #
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(256, 128), # Linear transformation part input*W+b
        #     nn.LeakyReLU(0.2), # leaky relu is more robust for GANs than ReLU
        #     nn.Dropout(dropout)
        # )

        self.out = nn.Sequential(
            nn.Linear(hidden_layers[-1], 1),
        )

    # Careful to make forward() a function of the net, not of __init__
    def forward(self, x):
        for i_hidden in range(len(self.hidden)):
            x = self.hidden[i_hidden](x)
        x = self.out(x)
        return x


def interpolate_data(real_data, fake_data):
    eps = torch.rand(real_data.size(0), 1) # A random unif number for each obs in the batch
    eps = eps.expand(real_data.size()) # Can only multiply tensors with tensors, so expand to same dimensions
    interpolated_data = eps*real_data + (1-eps)*fake_data
    interpolated_data = Variable(interpolated_data, requires_grad=True) # Transform into Variable again
    return interpolated_data

def calc_gradient_penalty(critic, real_data, fake_data):
    interpolated_data = interpolate_data(real_data, fake_data)
    critic_output = critic(interpolated_data)
    gradients = autograd.grad(inputs=interpolated_data, outputs=critic_output,
                             grad_outputs=torch.ones(critic_output.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty=((gradients.norm(2, dim=1)-1) ** 2)
    return gradient_penalty


# def critic_loss_GP(critic, real_data, fake_data, penalty_coefficient):
#     """
#     Wasserstein distance to minimize as loss for the critic, regularized by
#     Lipschwitz 1 gradient penalty
#     -(E[D(x_real)] - E[D(x_fake)]) + lambda*E[(||D(x_imputed)'||_2 -1)**2]
#     """
#     # Original critic loss
#     # D(x_real)
#     output_real = critic.forward(real_data)
#     # D(x_fake)
#     output_fake = critic.forward(fake_data)
#     raw_loss = (output_fake - output_real).squeeze()

#     # Gradient penalty for Lipschwitz-1
#     gradient_penalty = calc_gradient_penalty(critic, real_data,fake_data)

#     # Total loss
#     loss = (raw_loss + penalty_coefficient * gradient_penalty).mean()
#     return loss

def critic_loss(output_real, output_fake):
    """
    Wasserstein distance to minimize as loss for the critic
    -(E[D(x_real)] - E[D(x_fake)])
    """
    return -( torch.mean(output_real) - torch.mean(output_fake) )

def generator_loss(output_fake):
    """
    Loss to minimize for the generator on the output of the optimal critic
    -E[D(G(noise))]
    """
    return -torch.mean(output_fake)


# def train_critic(optimizer, real_data, fake_data, gradient_penalty_coefficient=10):
#     N = real_data.size(0) # Get number of rows from torch tensor
#     optimizer.zero_grad() # reset gradient
#
#     # Note: Calling backward() multiple times will acumulate the gradients
#     # until they are reset with zero_grad()
#     # E[D(x_real)]
#     output_real = critic.forward(real_data)
#
#     # E[D(x_fake)]
#     output_fake = critic.forward(fake_data)
#     raw_loss = critic_loss(output_real, output_fake)
#
#     # Gradient penalty
#     gradient_penalty = calc_gradient_penalty(critic, real_data,fake_data)
#
#     # Calculate overall loss
#     # Minimize the raw loss pushed upwards by penalty (always positive)
#     loss = raw_loss + gradient_penalty_coefficient*gradient_penalty
#     loss = loss.mean() # Average over batch
#
#     # Weight update
#     loss.backward()
#     optimizer.step()
#
#     # Return error and predictions for monitoring
#     return raw_loss.mean(), output_real, output_fake
#
#
# def train_generator(optimizer, fake_data):
#     N = fake_data.size(0) # Get number of rows from torch tensor
#     optimizer.zero_grad() # reset gradient
#
#     # Get discriminator prediction output
#     critic_prediction = critic.forward(fake_data)
#
#     # See explanation above. Intuitively, we create loss if the
#     # discriminator predicts our pseudo-ones as zeros.
#     loss_generator = generator_loss(critic_prediction)
#     loss_generator.backward()
#
#     # Weight update
#     optimizer.step()
#
#     # Return error and predictions for monitoring
#     return loss_generator

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
            for n_batch, (real_batch,_) in enumerate(data_loader):
                N = real_batch.size(0) # Get the number of images from tensor

                ## Train discriminator
                # Collect real data
                real_data = Variable(real_batch.float())

                temp_performance = []
                for k in range(critic_rounds):
                    # Create fake data
                    fake_data = self.generator(make_noise(N, self.generator.noise_dim)).detach()
                    # generator() creates a graph on the fly, which we drop after collecting the fake data
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
                    if val_data is None:
                        print("{train:.6f} | ".format(
                            train=(self.critic(real_data).mean() - self.critic(self.generator(make_noise(N, self.generator.noise_dim))).mean()).detach().cpu().numpy()
                        ))
                    else:
                        print("{train:.6f} | {val:.6f}".format(
                            train=(self.critic(real_data).mean() - self.critic(self.generator(make_noise(N, self.generator.noise_dim))).mean()).detach().cpu().numpy(),
                            val  =(self.critic(val_data).mean()  - self.critic(self.generator(make_noise(val_data.size(0), self.generator.noise_dim))).mean()).detach().cpu().numpy()
                        ))

                    #print(pd.DataFrame(generator(validation_noise).detach().numpy()).mean())
            self.n_epoch += 1
        return critic_performance, generator_performance

    def train_critic(self, optimizer, real_data, fake_data, gradient_penalty_coefficient=10):
        N = real_data.size(0) # Get number of rows from torch tensor

        for param in self.critic.parameters():
            param.requires_grad = True
        optimizer.zero_grad() # reset gradient

        # Note: Calling backward() multiple times will acumulate the gradients
        # until they are reset with zero_grad()
        # E[D(x_real)]
        output_real = self.critic.forward(real_data)

        # E[D(x_fake)]
        output_fake = self.critic.forward(fake_data)
        raw_loss = critic_loss(output_real, output_fake)

        # Gradient penalty
        gradient_penalty = calc_gradient_penalty(self.critic, real_data,fake_data)

        # Calculate overall loss
        # Minimize the raw loss pushed upwards by penalty (always positive)
        loss = raw_loss + gradient_penalty_coefficient*gradient_penalty
        loss = loss.mean() # Average over batch

        # Weight update
        loss.backward()
        optimizer.step()

        # Return error and predictions for monitoring
        return raw_loss.mean(), output_real, output_fake

    def train_generator(self, optimizer, N):
        for param in self.critic.parameters():
            param.requires_grad = False
        optimizer.zero_grad() # reset gradient

        # Create fake data
        fake_data = self.generator(make_noise(N, dim = self.generator.noise_dim))

        # Get discriminator prediction output
        critic_prediction = self.critic.forward(fake_data)

        # See explanation above. Intuitively, we create loss if the
        # discriminator predicts our pseudo-ones as zeros.
        loss_generator = generator_loss(critic_prediction)
        loss_generator.backward()

        # Weight update
        optimizer.step()

        # Return error and predictions for monitoring
        return loss_generator
