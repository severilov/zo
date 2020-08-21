import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-paper')

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 12
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['figure.figsize'] = (10.0, 6.0)

digit_size = 14


def show_images(x, epoch=None):
    plt.figure(figsize=(12, 12 / 10 * (x.shape[0] // 10 + 1)))
    x = x.view(-1, digit_size, digit_size)
    for i in range(x.shape[0]):
        plt.subplot(x.shape[0] // 10 + 1, 10, i + 1)
        plt.imshow(x.data[i].cpu().numpy(), cmap='Greys_r', vmin=0, vmax=1, interpolation='lanczos')
        plt.axis('off')
        if epoch is not None:
            plt.savefig('./experiments/train_imgs' + epoch + '.png')


def plot_losses(G_losses, D_losses, optD, optG, save=False):
    plt.figure(figsize=(11, 7))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(np.arange(len(G_losses)) / 1875, G_losses, label="G", color='navy')
    plt.plot(np.arange(len(D_losses)) / 1875, D_losses, label="D", color='darkorange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    legend = plt.legend(loc='best', shadow=True, title=optD + ', ' + optG)
    frame = legend.get_frame()
    frame.set_facecolor('papayawhip')
    frame.set_edgecolor('red')
    plt.grid()
    if save:
        plt.savefig('./experiments/loss_' + optD + '_' + optG + '.png')

    plt.show()


def plot_ll(gan, optD, optG, save=False):
    plt.figure(figsize=(11, 7))
    '''
    for label, name, model in [
        ('zoVIA, zoVIA log-likelihoods', 'log_likelihoods', gan_zovia_adam),
        ('zoVIA, Adam log-likelihoods', 'log_likelihoods', gan_zovia_adam),
        ('Adam, RMSprop log-likelihoods', 'log_likelihoods', gan_zovia_adam),
    ]:
        data = model[name]
        x_labels = np.arange(len(data))
        plt.plot(x_labels, data, label=label)
    '''
    data = gan['log_likelihoods']
    x_labels = np.arange(len(data))
    plt.plot(x_labels, data, label=optD + ', ' + optG)
    plt.xlabel('Epoch')
    # plt.xlim(xmin=0.0, xmax=x_labels[-1])
    plt.ylabel('Log-likelihood estimation')
    plt.grid(True)
    plt.legend(loc='best')

    if save:
        plt.savefig('./experiments/likelihood_' + optD + '_' + optG + '.png')
    plt.show()
