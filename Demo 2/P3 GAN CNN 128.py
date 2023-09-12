import torch
import torch.nn as nn

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import time

# Constants
leaky_relu_slope = 0.2


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            self._block(features_d * 1, features_d * 2, 4, 2, 1),  # 32x32 -> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16 -> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8 -> 4x4
            nn.Conv2d(
                features_d * 8, 1, kernel_size=4, stride=2, padding=0
            ),  # 2x2 -> 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Defines block
        """

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # Since we're using batchnorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 2, 0),  # N x f_g*16 x 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # N x f_g*8 x 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # N x f_g*4 x 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # N x f_g*2 x 32 x 32
            self._block(features_g * 2, features_g * 1, 4, 2, 1),  # N x f_g*2 x 32 x 32
            nn.ConvTranspose2d(
                features_g, img_channels, kernel_size=4, stride=2, padding=1
            ),  # N x img_channels x 128 x 128
            nn.Tanh(),  # Normalizes the output between -1 and 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Upsacpe. use nn.ConvTranspose2D. Does opposite of conv layer
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # Since we're using batchnorm2d
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # use relu activation function
        )

    def forward(self, x):
        return self.gen(x)


# Initialise weights
def init_weights(model):
    # mean 0, sd =0.2
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Hyperparameters

LEARNING_RATE = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 128
FEATURES_GEN = 128

GEN_BATCH_SIZE = 32

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")


def test():
    TEST_BATCH_SIZE = 8
    H, W = IMAGE_SIZE, IMAGE_SIZE

    # Generate random noise
    x = torch.randn((TEST_BATCH_SIZE, CHANNELS_IMG, H, W))

    # Discriminator
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC)
    init_weights(disc)

    print("Discriminator shape: ", disc(x).shape)

    assert disc(x).shape == (TEST_BATCH_SIZE, 1, CHANNELS_IMG, CHANNELS_IMG)
    print("Discriminator valid")

    # Generator
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)

    # Initialise weights
    init_weights(gen)

    # generate latent noise
    z = torch.randn((TEST_BATCH_SIZE, Z_DIM, CHANNELS_IMG, CHANNELS_IMG))

    print("Generator shape: ", gen(z).shape, " = ", TEST_BATCH_SIZE, CHANNELS_IMG, H, W)
    assert gen(z).shape == (TEST_BATCH_SIZE, CHANNELS_IMG, H, W)
    print("Generator valid")


# test()
# exit()

# Storing losses for plotting
disc_losses = []
gen_losses = []


# Transforms

transforms = transforms.Compose(
    [
        transforms.CenterCrop(
            178
        ),  # Crop the central 178x178 pixels (CelebA image size)
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

path = "R:\\COMP3710 Data\\"

# Output folder for saving images
# Define the folder path for saving generated images
output_folder = "R:\\COMP3710 Data\\generated_images"
os.makedirs(output_folder, exist_ok=True)

# Training data
dataset = torchvision.datasets.CelebA(
    root=path + "data/celeba",
    split="all",  # Use the 'all' split for both training and testing
    transform=transforms,
    download=False,  # Set to True if you haven't downloaded the dataset yet
)

# data loader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
# Discriminator
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

# Iniitialise weights
init_weights(gen)
init_weights(disc)

# Optimiser
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Loss
criterion = nn.BCELoss()

# Fixed noise
fixed_noise = torch.randn(GEN_BATCH_SIZE, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(path + "logs\\real")
# writer_fake = SummaryWriter(path + "logs\\fake")

step = 0

# Training
gen.train()
disc.train()


for epoch in range(NUM_EPOCHS):
    start_time = time.time()  # Record the start time of the epoch
    for batch_index, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)

        # Generate fakes
        fake = gen(noise)

        # Train discriminator. Maximise log(D(x)) + log(1 - D(G(z)))

        disc_real = disc(real).reshape(-1)  # N
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).reshape(-1)  # N
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train generator min log(1 - D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_index % 100 == 0:
            print(batch_index, end=", ")

        if batch_index % 500 == 0:
            print()
            with torch.no_grad():
                fake = gen(fixed_noise)

                # Save generated images to the output folder
                for i, generated_image in enumerate(fake):
                    image_filename = os.path.join(
                        output_folder,
                        f"epoch_{epoch}_batch_{batch_index}_image_{i}.png",
                    )
                    torchvision.utils.save_image(
                        generated_image, image_filename, normalize=True
                    )

                # fake = gen(fixed_noise)

                # # Display generated images
                # plt.figure(figsize=(8, 8))
                # plt.imshow(
                #     torchvision.utils.make_grid(fake[:GEN_BATCH_SIZE], normalize=True)
                #     .cpu()
                #     .numpy()
                #     .transpose(1, 2, 0)
                # )
                # plt.axis("off")
                # plt.show()

                # Display training progress
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_index}/{len(loader)}] "
                    f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )

    end_time = time.time()  # Record the end time of the epoch
    epoch_time = end_time - start_time  # Calculate the elapsed time for the epoch
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Time: {epoch_time:.2f} seconds")


# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(disc_losses, label="Discriminator Loss")
plt.plot(gen_losses, label="Generator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

with torch.no_grad():
    fake = gen(fixed_noise)
    plt.figure(figsize=(8, 8))
    plt.imshow(
        torchvision.utils.make_grid(fake[:GEN_BATCH_SIZE], normalize=True)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    plt.axis("off")
    plt.show()
