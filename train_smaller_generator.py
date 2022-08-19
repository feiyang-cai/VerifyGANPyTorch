from src.smaller_generator import SmallerGenerator
from src.gan import Generator, Discriminator
import torch.optim as optim
from utils import orthogonal_regularization, taxi_input, save_generator_image
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SmallerG = SmallerGenerator(2, 2).to(device)
SmallerG.train()

G = Generator(2, 2).to(device)
G.load_state_dict(torch.load("./outputs/generator.pth", map_location=device))
G.eval()

D = Discriminator(2).to(device)
D.load_state_dict(torch.load("./outputs/discriminator.pth", map_location=device))
D.eval()

lr = 1e-3
lam = 7e-3
num_epochs = 50000
verbose_freqency = 1000

optimizer = optim.Adam(SmallerG.parameters(), lr=lr)
criterion = nn.L1Loss().to(device)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    z, y = taxi_input(b_size=256, device=device)
    x_tilde = SmallerG(z, y)
    with torch.no_grad():
        x = G(z, y)
    loss = criterion(x, x_tilde) - \
           lam * torch.mean(torch.tanh(D(x_tilde, y))) + \
           orthogonal_regularization(SmallerG, device)
    loss.backward()
    optimizer.step()


    if epoch % verbose_freqency == 0:
        print(f"Epoch {epoch+1} of {num_epochs}")
        print(f"Loss: {loss:.8f}")
        # save the generated torch tensor models to disk
        z, y = taxi_input(b_size=16, device=device)
        with torch.no_grad():
            x_tilde = SmallerG(z, y)
        save_generator_image(x_tilde, f"./smaller_outputs/gen_img{epoch}.png")

print('DONE TRAINING')
# save the model weights to disk
torch.save(SmallerG.state_dict(), './smaller_outputs/smaller_generator.pth')
    
