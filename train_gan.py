from src.gan import Generator, Discriminator
from src.taxi_data import TaxiDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import orthogonal_regularization, save_generator_image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


G = Generator(2, 2).to(device)
D = Discriminator(2).to(device)

dataset = TaxiDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                         shuffle=True, num_workers=8)

optimizerD = optim.Adam(D.parameters(), lr=7e-4, betas=(0.5, 0.99))
optimizerG = optim.Adam(G.parameters(), lr=7e-4, betas=(0.5, 0.99))
criterion = nn.BCEWithLogitsLoss().to(device)

G.train()
D.train()

def label_real(size):
    """
    Fucntion to create real labels (ones)
    :param size: batch size
    :return real label vector
    """
    data = torch.ones(size, 1)
    return data.to(device)

def label_fake(size):
    """
    Fucntion to create fake labels (zeros)
    :param size: batch size
    :returns fake label vector
    """
    data = torch.zeros(size, 1)
    return data.to(device)


def train_generator(optimizer, data_fake, y_fake):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    
    optimizer.zero_grad()

    output = D(data_fake, y_fake)
    loss = criterion(output, real_label)
    loss += orthogonal_regularization(G, device)

    loss.backward()

    optimizer.step()
    return loss

def train_discriminator(optimizer, data_real, data_fake, y_real, y_fake):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)

    optimizer.zero_grad()
    
    output_real = D(data_real, y_real)
    loss_real = criterion(output_real, real_label)

    output_fake = D(data_fake, y_fake)
    loss_fake = criterion(output_fake, fake_label)

    loss_real.backward()
    loss_fake.backward()

    optimizer.step()

    return loss_real + loss_fake



print("Starting Training Loop...")
num_epochs = 750
losses_G = []
losses_D = []

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    loss_G, loss_D = 0.0, 0.0
    for i, (x,y) in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        b_size = x.size(0)
        x, y = x.to(device), y.to(device)

        # Train the discriminator 
        z_noise = Variable(FloatTensor(np.random.uniform(-1.0, 1.0, size=(b_size, 2))))
        y_noise = Variable(FloatTensor(np.concatenate([np.random.uniform(-1.73, 1.73, size=(b_size, 1)), 
                                                        np.random.uniform(-1.74, 1.74, size=(b_size, 1))], axis=1)))
        data_fake = G(z_noise, y_noise).detach()
        data_real = x
        loss_D += train_discriminator(optimizerD, data_real, data_fake, y, y_noise)

        # Train the generator
        z_noise = Variable(FloatTensor(np.random.uniform(-1.0, 1.0, size=(b_size, 2))))
        y_noise = Variable(FloatTensor(np.concatenate([np.random.uniform(-1.73, 1.73, size=(b_size, 1)), 
                                                        np.random.uniform(-1.74, 1.74, size=(b_size, 1))], axis=1)))
        data_fake = G(z_noise, y_noise)
        loss_G += train_generator(optimizerG, data_fake, y_noise)

    z_noise = Variable(FloatTensor(np.random.uniform(-1.0, 1.0, size=(b_size, 2))))
    y_noise = Variable(FloatTensor(np.concatenate([np.random.uniform(-1.73, 1.73, size=(b_size, 1)), 
                                                    np.random.uniform(-1.74, 1.74, size=(b_size, 1))], axis=1)))
    generated_img = G(z_noise, y_noise).cpu().detach()
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"./outputs/gen_img{epoch}.png")
    epoch_loss_g = loss_G / i # total generator loss for the epoch
    epoch_loss_d = loss_D / i # total discriminator loss for the epoch
    #losses_G.append(epoch_loss_g)
    #losses_D.append(epoch_loss_d)

    print(f"Epoch {epoch+1} of {num_epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

print('DONE TRAINING')
# save the model weights to disk
torch.save(G.state_dict(), './outputs/generator.pth')
torch.save(D.state_dict(), './outputs/discriminator.pth')

# plot and save the generator and discriminator loss
#plt.figure()
#plt.plot(losses_G, label='Generator loss')
#plt.plot(losses_D, label='Discriminator Loss')
#plt.legend()
#plt.savefig('./outputs/loss.png')


