import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class WGAN():
    def __init__(self, generator, critic, G_optimizer, C_optimizer,
                 gp_weight=10, critic_iterations=5, verbose=0, print_every=100,
                 use_cuda=False):
        #TODO: Standardize names to generator/critic
        self.G = generator
        self.G_opt = G_optimizer
        self.D = critic
        self.D_opt = C_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], "distance":[], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.verbose = verbose
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data, aux_data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()

        # Generate fake data
        generated_data = self.sample_generator(batch_size, aux_data).detach()

        self.D_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.D(data,aux_data)
        d_generated = self.D(generated_data, aux_data)

        # Get gradient penalty
        gradient_penalty = self.gp_weight*self._gradient_penalty(data, generated_data, aux_data)

        # Create total loss and optimize
        d_distance = d_real.mean() - d_generated.mean()
        # The Wasserstein distance is the supremum (maximum) between the two expectations
        # in order to maximize, we minimize the negative loss and
        # add the penalty
        d_loss = -d_distance + gradient_penalty

        d_loss.backward()
        self.D_opt.step()

        # Record loss
        if self.verbose > 1:
            if i % self.print_every == 0:
                self.losses['GP'].append(gradient_penalty.data)
                self.losses['D'].append(-d_distance.data)
                self.losses["distance"].append(d_distance.data)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, aux_data)

        # Calculate loss and optimize
        d_generated = self.D(generated_data, aux_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G.training_iterations += 1

        if self.verbose > 1:
            if i % self.print_every == 0:
                self.losses['G'].append(g_loss.data)

    def _gradient_penalty(self, real_data, generated_data, aux_data):
        assert real_data.size() == generated_data.size(), f'real and generated mini batches must have same size ({real_data.size()} and {generated_data.size()})'
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, *[1 for _ in range(real_data.dim()-1)])
        #alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolated = alpha * real_data.data + (1. - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate distance of interpolated examples
        d_interpolated = self.D(interpolated, aux_x=aux_data)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(d_interpolated.size()).cuda()
                               if self.use_cuda else torch.ones(
                                      d_interpolated.size()),
                                      create_graph=True, retain_graph=True,
                                      only_inputs=True
                               )[0]

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        if self.verbose > 0:
            if i % self.print_every == 0:
                self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)
        return ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            if self.G.training_iterations < 10:
                critic_iterations = self.critic_iterations * 5
            else:
                critic_iterations = self.critic_iterations

            self.num_steps += 1
            if len(data[1].shape) == 1:
                data[1] = data[1].unsqueeze(dim=1)

            self._critic_train_iteration(data[0], data[1])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % critic_iterations == 0:
                self._generator_train_iteration(data[0], data[1])

            if self.verbose > 1:
                if i % self.print_every == 0:
                    print("Iteration {}".format(i + 1))
                    print("D: {}".format(self.losses['D'][-1]))
                    print("GP: {}".format(self.losses['GP'][-1]))
                    print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                    if self.num_steps > critic_iterations:
                        print("G: {}".format(self.losses['G'][-1]))
                        print("Distance: {}".format(self.losses['distance'][-1]))

    def train(self, data_loader, epochs, save_training_gif=False):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

        #     if save_training_gif:
        #         # Generate batch of images and convert to grid
        #         img_grid = make_grid(self.G(fixed_latents).cpu().data)
        #         # Convert to numpy and transpose axes to fit imageio convention
        #         # i.e. (width, height, channels)
        #         img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
        #         # Add image grid to training progress
        #         training_progress_images.append(img_grid)
        #
        # if save_training_gif:
        #     imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
        #                     training_progress_images)

    def sample_generator(self, num_samples, aux_data=None):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        if self.G.aux_dim == 0:
            generated_data = self.G(latent_samples)
        else:
            generated_data = self.G(latent_samples, aux_data)
        return generated_data

    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     # Remove color channel
    #     return generated_data.data.cpu().numpy()[:, 0, :, :]
