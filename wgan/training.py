"""
Classes for different GAN architectures
VanillaGAN : Goodfellow, Ouget, Mirza, Xu & Warde-Farley (2014) Generative Adversarial Nets
WassersteinGAN : Arjovsky, Chintala & Bottou (2017) Wasserstein GAN
FisherGAN : Mroueh & Sercu (2017) Fisher GAN
"""

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class VanillaGAN():
    """
    Vanilla GAN and base class for different GAN architectures.

    Defines functions train(), _train_epoch() and sample_generator()

    Attributes:
        num_steps (int): The number of weight updates (critic). Generator
                         updates are num_steps/critic_iterations
    """
    def __init__(self, generator, critic, g_optimizer, c_optimizer,
                 critic_iterations=5, verbose=0, print_every=100,
                 use_cuda=False):
        """
        Args:
            generator : pytorch nn module
              A pytorch module that outputs artificial data samples

            critic: pytorch nn module
              A pytorch module that discriminates between real and fake data

            g_optimizer, c_optimizer : pytorch optimizer object

            critic_iterations : int

            verbose : int

            print_every : int

            use_code : bool
        """
        self.use_cuda = use_cuda
        self.generator = generator
        self.g_opt = g_optimizer

        self.critic = critic
        self.c_opt = c_optimizer
        self.critic_iterations = critic_iterations

        # Monitoring
        self.num_steps = 0
        self.verbose = verbose
        self.print_every = print_every
        self.losses = {'G': [], 'D': []}

        if self.use_cuda:
            self.generator.cuda()
            self.critic.cuda()

    def _critic_train_iteration(self, data, aux_data):
        """ """
        # Get data
        batch_size = data.size()[0]
        data = Variable(data)
        aux_data = Variable(aux_data)
        ones = torch.ones(batch_size)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
            aux_data = aux_data.cuda()
            ones = ones.cuda()

        # Generate fake data
        generated_data = self.sample_generator(batch_size, aux_data).detach()

        self.c_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.critic(data, aux_data)
        d_generated = self.critic(generated_data, aux_data)

        # Create total loss and optimize
        d_distance = d_real.log().mean() + (ones - d_generated).log().mean()
        # We do gradient descent, not ascent as described in the original paper
        c_loss = -d_distance

        c_loss.backward()
        self.c_opt.step()
        self.critic.training_iterations += 1

        # Record loss
        if self.verbose > 1:
            if self.generator.training_iterations % self.print_every == 0 and \
               self.critic.training_iterations % self.print_every == 0:
                self.losses['D'].append(-d_distance.data)
                self.losses["distance"].append(d_distance.data)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, aux_data)
        ones = torch.ones(batch_size)
        if self.use_cuda:
            ones.cuda()

        # Calculate loss and optimize
        d_generated = self.critic(generated_data, aux_data)
        g_loss = (ones - d_generated).mean()
        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.generator.training_iterations += 1

        if self.verbose > 1:
            if self.generator.training_iterations % self.print_every == 0 and \
               self.critic.training_iterations % self.print_every == 0:
                self.losses['G'].append(g_loss.data)

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            if self.generator.training_iterations < 10:
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

            if self.verbose > 2:
                if i % self.print_every == 0:
                    print("Iteration {}".format(i + 1))
                    print("D: {}".format(self.losses['D'][-1]))
                    if self.num_steps > critic_iterations:
                        print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=False):
        """
        Train the GAN generator and critic on the data given by the data_loader. GAN
        needs to be trained before synthetic data can be created.

        Arguments
        ---------
        data_loader :

        epochs : int
        Number of runs through the data (epochs)
        """
        if save_training_gif:
            Warning("save_training_gif not implemented. Ignored.")
        #     # Fix latents to see how image generation improves during training
        #     fixed_latents = Variable(self.generator.sample_latent(64))
        #     if self.use_cuda:
        #         fixed_latents = fixed_latents.cuda()
        #     training_progress_images = []

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

        #     if save_training_gif:
        #         # Generate batch of images and convert to grid
        #         img_grid = make_grid(self.generator(fixed_latents).cpu().data)
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
        """
        Generate num_samples observations from the generator.

        Arguments
        ---------
        num_samples: int
        Number of observations to generate

        aux_data: object
        Auxiliary data in the format used to train the generator
        """
        latent_samples = Variable(self.generator.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        if self.generator.aux_dim == 0:
            generated_data = self.generator(latent_samples)
        else:
            generated_data = self.generator(latent_samples, aux_data)
        return generated_data

    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     # Remove color channel
    #     return generated_data.data.cpu().numpy()[:, 0, :, :]

    #TODO: At function to plot loss during training

class WassersteinGAN(VanillaGAN):
    """
    Class for Critic GAN with Wasserstein loss function
    """
    def __init__(self, generator, critic, g_optimizer, c_optimizer,
                 gp_weight=10, critic_iterations=5, verbose=0, print_every=100,
                 use_cuda=False):
        super().__init__(generator=generator, critic=critic,
                         g_optimizer=g_optimizer, c_optimizer=c_optimizer,
                         critic_iterations=critic_iterations,
                         verbose=verbose, print_every=print_every,
                         use_cuda=use_cuda)

        self.gp_weight = gp_weight

        # Monitoring
        self.losses = {'G': [], 'D': [], 'GP': [], "distance":[], 'gradient_norm': []}

        if self.use_cuda:
            self.generator.cuda()
            self.critic.cuda()

    def _critic_train_iteration(self, data, aux_data):
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

        self.c_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.critic(data, aux_data)
        d_generated = self.critic(generated_data, aux_data)

        # Get gradient penalty
        gradient_penalty = self.gp_weight*self._gradient_penalty(data, generated_data, aux_data)

        # Create total loss and optimize
        d_distance = d_real.mean() - d_generated.mean()
        # The Wasserstein distance is the supremum (maximum) between the two expectations
        # in order to maximize, we minimize the negative loss and
        # add the penalty
        c_loss = -d_distance + gradient_penalty

        c_loss.backward()
        self.c_opt.step()
        self.critic.training_iterations += 1

        # Record loss
        if self.verbose > 1:
            if self.generator.training_iterations % self.print_every == 0 and \
               self.critic.training_iterations % self.print_every == 0:
                self.losses['GP'].append(gradient_penalty.data)
                self.losses['D'].append(-d_distance.data)
                self.losses["distance"].append(d_distance.data)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, aux_data)

        # Calculate loss and optimize
        d_generated = self.critic(generated_data, aux_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.generator.training_iterations += 1

        if self.verbose > 1:
            if self.generator.training_iterations % self.print_every == 0 and \
               self.critic.training_iterations % self.print_every == 0:
                self.losses['G'].append(g_loss.data)

    def _gradient_penalty(self, real_data, generated_data, aux_data):
        assert real_data.size() == generated_data.size(), ('real and generated mini batches must '
                                                           'have same size ({a} and {b})').format(
                                                               a=real_data.size(),
                                                               b=generated_data.size())
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
        d_interpolated = self.critic(interpolated, aux_x=aux_data)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=d_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(d_interpolated.size()).cuda()
                               if self.use_cuda else torch.ones(d_interpolated.size()),
                               create_graph=True, retain_graph=True,
                               only_inputs=True
                               )[0]

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        ## Return gradient penalty
        #if self.verbose > 0:
        #    if i % self.print_every == 0:
        #        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)
        return ((gradients_norm - 1) ** 2).mean()


class FisherGAN(VanillaGAN):
    """
    Class for GAN with Fisher loss function
    """
    def __init__(self, generator, critic, g_optimizer, c_optimizer,
                 critic_iterations=5, penalty=1e-6, verbose=0, print_every=100,
                 use_cuda=False):
        super().__init__(generator=generator, critic=critic,
                         g_optimizer=g_optimizer, c_optimizer=c_optimizer,
                         critic_iterations=critic_iterations,
                         verbose=verbose, print_every=print_every,
                         use_cuda=use_cuda)

        # Monitoring
        self.losses = {'G': [], 'D': [], "distance":[], 'lagrange_multiplier': []}

        self.penalty = penalty
        self.lagrange_mult = torch.FloatTensor([0])
        self.lagrange_mult = Variable(self.lagrange_mult, requires_grad=True)

        if self.use_cuda:
            self.generator.cuda()
            self.critic.cuda()
            self.lagrange_mult.cuda()
            self.penalty.cuda()

    def _critic_train_iteration(self, data, aux_data):
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

        self.c_opt.zero_grad()
        # Calculate critic output of real and fake data
        d_real = self.critic(data, aux_data)
        d_generated = self.critic(generated_data, aux_data)
        distance = d_real.mean() - d_generated.mean()

        # Calculate constraint \Omega
        constraint = 1 - (0.5*(d_real**2).mean() + 0.5*(d_generated**2).mean())

        # Create total loss and optimize
        c_loss = distance + self.lagrange_mult * constraint - self.penalty/2 * constraint**2

        # Maximize critic weights w.r.t loss
        (-c_loss).backward()
        self.c_opt.step()
        self.critic.training_iterations += 1

        # Minimize lagrange multiplier lambda w.r.t to loss and quadratic penalty rho
        self.lagrange_mult.data += self.penalty * self.lagrange_mult.grad.data
        self.lagrange_mult.grad.data.zero_()

        # Record loss
        if self.verbose > 1:
            if self.generator.training_iterations % self.print_every == 0 and \
               self.critic.training_iterations % self.print_every == 0:
                self.losses['lagrange_multiplier'].append(self.lagrange_mult.data)
                self.losses['D'].append(c_loss.data)
                self.losses["distance"].append(distance.data)

    def _generator_train_iteration(self, data, aux_data):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, aux_data)

        # Calculate loss and optimize
        d_generated = self.critic(generated_data, aux_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.generator.training_iterations += 1

        if self.verbose > 1:
            if self.generator.training_iterations % self.print_every == 0 and \
               self.critic.training_iterations % self.print_every == 0:
                self.losses['G'].append(g_loss.data)
