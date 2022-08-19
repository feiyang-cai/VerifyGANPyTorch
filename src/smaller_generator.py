import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class SmallerGenerator(nn.Module):
    def __init__(self, nclasses, latent_dim):
        super(SmallerGenerator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(nclasses+latent_dim, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 8*16),
        )

        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.orthogonal_(m.weight.data)

    def forward(self, x, y):
        t = self.main(torch.cat([x, y], dim=1))
        t = t.reshape(-1, 1, 8, 16)
        return t

if __name__ == "__main__":
    G = SmallerGenerator(2, 2)

    x = torch.randn(128, 2)
    y = torch.randn(128, 2)
    print(G(x, y).shape)
