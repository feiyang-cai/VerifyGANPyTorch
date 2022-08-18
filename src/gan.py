import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class Generator(nn.Module):
    def __init__(self, nclasses, latent_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(nclasses, 32) 
        self.fc2 = nn.Linear(latent_dim, 992)

        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),

            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.orthogonal_(m.weight.data)

    def forward(self, z, y):
        y = self.fc1(y)
        y = y.reshape(-1, 16, 1, 2)
        z = self.fc2(z)
        z = z.reshape(-1, 496, 1, 2)
        inputs = torch.cat([y, z], dim=1)
        t = self.main(inputs)
        return t

class Discriminator(nn.Module):
    def __init__(self, nclasses):
        super(Discriminator, self).__init__()
        self.fc = spectral_norm(nn.Linear(nclasses, 16*8))

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 3, 1, 1)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.1),

            spectral_norm(nn.Conv2d(512, 512, 3, 1, 1)),
            nn.LeakyReLU(0.1),

            nn.Flatten(1, -1),
            nn.Linear(1024, 1),
            #nn.Sigmoid()
        )

        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x, y):
        y = self.fc(y)
        y = y.reshape(-1, 1, 8, 16)
        inputs = torch.cat([y, x], dim=1)
        t = self.main(inputs)
        return t

"""
class Discriminator(nn.Module):
    def __init__(self, nclasses):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(nclasses, 16*8)

        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Flatten(1, -1),
            nn.Linear(1024, 1),
            #nn.Sigmoid()
        )

        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x, y):
        y = self.fc(y)
        y = y.reshape(-1, 1, 8, 16)
        inputs = torch.cat([y, x], dim=1)
        t = self.main(inputs)
        return t
"""


    

if __name__ == "__main__":
    G = Generator(2, 2)
    y = torch.randn(128, 2)
    x = torch.randn(128, 2)
    G(y, x)

    D = Discriminator(2)
    y = torch.randn(128, 2)
    x = torch.randn(128, 1, 8, 16)
    D(y, x)