# train.py (updated with dynamic label shape and model saving)

import torch
import itertools
import os
from config import Config
from models import Generator, Discriminator, weights_init_normal
from data_loader import get_dataloaders
from torchvision.utils import save_image


def train():
    loader_A, loader_B = get_dataloaders()

    G_AB = Generator().to(Config.DEVICE)
    G_BA = Generator().to(Config.DEVICE)
    D_A = Discriminator().to(Config.DEVICE)
    D_B = Discriminator().to(Config.DEVICE)

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Losses
    MSE = torch.nn.MSELoss()
    L1 = torch.nn.L1Loss()

    # Optimizers
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=Config.LR, betas=(Config.BETA1, Config.BETA2))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=Config.LR, betas=(Config.BETA1, Config.BETA2))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=Config.LR, betas=(Config.BETA1, Config.BETA2))

    for epoch in range(Config.NUM_EPOCHS):
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A = real_A.to(Config.DEVICE)
            real_B = real_B.to(Config.DEVICE)

            # Generate labels based on discriminator output size
            valid = torch.ones_like(D_A(real_A))
            fake = torch.zeros_like(D_A(real_A))

            # --- Train Generators ---
            opt_G.zero_grad()

            fake_B = G_AB(real_A)
            recov_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            recov_B = G_AB(fake_A)

            loss_id_A = L1(G_BA(real_A), real_A) * Config.LAMBDA_ID
            loss_id_B = L1(G_AB(real_B), real_B) * Config.LAMBDA_ID

            loss_GAN_AB = MSE(D_B(fake_B), valid)
            loss_GAN_BA = MSE(D_A(fake_A), valid)

            loss_cycle_A = L1(recov_A, real_A) * Config.LAMBDA_CYCLE
            loss_cycle_B = L1(recov_B, real_B) * Config.LAMBDA_CYCLE

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            opt_G.step()

            # --- Train Discriminator A ---
            opt_D_A.zero_grad()
            loss_D_A_real = MSE(D_A(real_A), valid)
            loss_D_A_fake = MSE(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            # --- Train Discriminator B ---
            opt_D_B.zero_grad()
            loss_D_B_real = MSE(D_B(real_B), valid)
            loss_D_B_fake = MSE(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{Config.NUM_EPOCHS}] [Batch {i}] "
                      f"[D_A: {loss_D_A.item():.4f}, D_B: {loss_D_B.item():.4f}] "
                      f"[G: {loss_G.item():.4f}]")

        os.makedirs("outputs", exist_ok=True)
        save_image((fake_B * 0.5 + 0.5), f"outputs/fakeB_epoch{epoch}.png")

    # Save final generator models
    torch.save(G_AB.state_dict(), "generator_photo2vangogh.pth")
    torch.save(G_BA.state_dict(), "generator_vangogh2photo.pth")


if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    train()
