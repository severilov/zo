import os
import pickle
import time
from torch import optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from models import Discriminator, Generator, device
from zo_opt import GradientEstimate_dicrs, GradientEstimate, zoVIA, zoESVIA, zoscESVIA
from log_likelihood import log_likelihood
from plot import *
from utils import *

train_data, valid_data, test_data = get_data()
real_label = 1.
fake_label = 0.
nz = 100  # Size of z latent vector (i.e. size of generator input)
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
print('Device: {}'.format(device))

def choose_optimizer(discriminator, generator, netD, netG, lr_d=2e-4, lr_g=2e-3):
    """
    Set optimizers for discriminator and generator
    :param discriminator: str, name
    :param generator: str, name
    :param netD:
    :param netG:
    :param lr_d:
    :param lr_g:
    :return: optimizerD, optimizerG
    """
    if discriminator == 'Adam':
        optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
    elif discriminator == 'RMSprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr=lr_d)
    elif discriminator == 'zoVIA':
        optimizerD = zoVIA(netD, lr=lr_d)
    elif discriminator == 'zoESVIA':
        optimizerD = zoESVIA(netD, lr=lr_d)
    elif discriminator == 'zoscESVIA':
        optimizerD = zoscESVIA(netD, lr=lr_d)

    if generator == 'Adam':
        optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    elif generator == 'RMSprop':
        optimizerG = optim.RMSprop(netG.parameters(), lr=lr_g)
    elif generator == 'zoVIA':
        optimizerG = zoVIA(netG, lr=lr_g)
    elif generator == 'zoESVIA':
        optimizerG = zoESVIA(netG, lr=lr_g)
    elif generator == 'zoscESVIA':
        optimizerG = zoscESVIA(netG, lr=lr_g)

    print('Discriminator optimizer: {}, lr={}'.format(discriminator, lr_d))
    print('Generator optimizer: {}, lr={}'.format(generator, lr_g))

    return optimizerD, optimizerG


def train_model(valid_data, test_data, dataloader,
                netD, netG, optimizerD, optimizerG,
                num_epochs=10, discr_zo=False, gener_zo=False,
                batch_size=32, tau=0.000001):
    """
    Train GAN function

    :param valid_data:
    :param test_data:
    :param dataloader:
    :param netD: Discriminator network
    :param netG: Generator network
    :param optimizerD: Discriminator optimizer
    :param optimizerG: Generator optimizer
    :param num_epochs:
    :param discr_zo: Discriminator optimizer Zero-order, bool
    :param gener_zo: Generator optimizer Zero-order, bool
    :param batch_size:
    :param tau:
    :return: gan, img_list
    """
    EPOCH_ZO_D = -1
    EPOCH_ZO_G = -1
    img_list = []
    G_losses, D_losses = [], []
    log_likelihoods = []
    iters = 0
    criterion = nn.BCELoss()

    print("Starting Training Loop...")

    generated_samples = generate_many_samples(netG, 512, batch_size).detach().cpu()
    valid_samples = valid_data[np.random.choice(len(valid_data), 512, False)][0]
    # valid_samples = valid_samples.to(next(model.parameters()).device)
    test_samples = test_data[np.random.choice(len(test_data), 512, False)][0]
    # test_samples = test_samples.to(next(model.parameters()).device)
    ll = log_likelihood(generated_samples, valid_samples, test_samples)
    # ll = 1
    log_likelihoods.append(ll)
    print('Log-likelihood before training: ', ll, flush=True)

    # For each epoch
    for epoch in range(num_epochs):
        print('EPOCH #{}'.format(epoch+1))
        start_epoch = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            if discr_zo and epoch > EPOCH_ZO_D:
                gradsD_real = GradientEstimate_dicrs(netD, real_cpu, label, criterion, tau)
                D_x = output.mean().item()
            elif discr_zo and epoch <= EPOCH_ZO_D:
                optimizerD_01ep = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
                errD_real.backward()
                D_x = output.mean().item()
            else:
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            if discr_zo and epoch > EPOCH_ZO_D:
                gradsD_fake = GradientEstimate_dicrs(netD, fake.detach(), label, criterion, tau)
                # print(grads_g)
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                gradsD = gradsD_real + gradsD_fake
                data = []
                optimizerD.step_update(netD, data, gradsD)
            elif discr_zo and epoch <= EPOCH_ZO_D:
                optimizerD_01ep = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD_01ep.step()
            else:
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            if gener_zo and epoch > EPOCH_ZO_G:
                # if gener_zo:
                # grads_g = GradientEstimate(netD, fake, label, criterion)
                grads_g = GradientEstimate(netG, netD, noise, label, criterion, tau)
                D_G_z2 = output.mean().item()
                data = []
                optimizerG.step_update(netG, data, grads_g)
            elif gener_zo and epoch <= EPOCH_ZO_G:
                optimizerG_01ep = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG_01ep.step()
            else:
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

            # Output training stats
            if i % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        print('EPOCH #{} time:'.format(epoch+1))
        timer(start_epoch, time.time())

        generated_samples = generate_many_samples(netG, 512, batch_size).detach().cpu()
        valid_samples = valid_data[np.random.choice(len(valid_data), 512, False)][0]
        # valid_samples = valid_samples.to(next(model.parameters()).device)
        test_samples = test_data[np.random.choice(len(test_data), 512, False)][0]
        # test_samples = test_samples.to(next(model.parameters()).device)
        ll = log_likelihood(generated_samples, valid_samples, test_samples)
        # ll = 1
        log_likelihoods.append(ll)
        print('Log-likelihood {} for epoch #{} \n'.format(ll, epoch+1))

    return {
               'netDiscriminator': netD.cpu(),
               'netGenerator': netG.cpu(),
               'generator_losses': G_losses,
               'discriminator_losses': D_losses,
               'log_likelihoods': log_likelihoods
           }, img_list


def main(optD, optG, num_epochs=5,
         discr_zo=False, gener_zo=False, save=True,
         tau=0.000001, lr_d=2e-4, lr_g=2e-3):
    """
    Make main experiment
    :param optD:
    :param optG:
    :param num_epochs:
    :param discr_zo:
    :param gener_zo:
    :param save:
    :param tau:
    :param lr_d:
    :param lr_g:
    :return: gan, img_list
    """
    if not os.path.exists('./experiments/'):
        os.makedirs('./experiments/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    # print(netG, netD)

    batch_size = 32
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizerD, optimizerG = choose_optimizer(optD, optG, netD, netG,
                                              lr_d=lr_d, lr_g=lr_g)
    gan, img_list = train_model(valid_data, test_data, dataloader,
                                netD, netG, optimizerD, optimizerG,
                                num_epochs=num_epochs, discr_zo=discr_zo,
                                gener_zo=gener_zo, batch_size=batch_size,
                                tau=tau)

    show_images(gan['netGenerator'].to(device).generate_samples(40))
    plot_losses(gan['generator_losses'], gan['discriminator_losses'], optD, optG, save=True)
    plot_ll(gan, optD, optG, save=True)

    if save:
        path = optD + '_' + optG + '_' + str(num_epochs) + 'ep.pickle'
        with open('./experiments/gan_' + path, 'wb') as handle:
            pickle.dump(gan, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Model saved at {}'.format('./experiments/gan_' + path))

        with open('./experiments/imgs_' + path, 'wb') as handle:
            pickle.dump(gan, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Control images for generator saved at {}'.format('./experiments/imgs_' + path))

    return gan, img_list