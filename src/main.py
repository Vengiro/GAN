import Generator
import Discriminator
import torchvision
import torch
import argparse
def main(epoch, batch_size, lr):
    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Should return True if CUDA is available
    print(torch.cuda.device_count())  # Number of GPUs detected
    print(torch.cuda.current_device())  # The ID of the current GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")
    train_data = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_labels = train_data.targets

    # Key advantage: - Data might be too large to fit in memory
    #                - Shuffle data and prevent overfit of batch patterns
    #                - Many functions available like: collate_fn, drop_last, num_workers, etc.
    #                - Next batch is loaded while current batch is being processed -> Faster training
    # [batch_size, channel_size, img_size, img_size]
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=100,
        shuffle=True
    )

    generator = Generator.Generator_MNIST().to(device)
    discriminator = Discriminator.Discriminator_MNIST().to(device)
    lossFunc = torch.nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    discr_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for i  in range(epoch):
        for j, (img, _) in enumerate(train_loader):
            img = img.to(device)
            noise = torch.randn(batch_size, 100).to(device)
            # Train Discriminator
            discr_optimizer.zero_grad()
            real_img = img
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)
            fake_img = generator.forward(noise)
            real_out = discriminator.forward(real_img)
            fake_out = discriminator.forward(fake_img)
            real_loss = lossFunc(real_out, real_label)
            fake_loss = lossFunc(fake_out, fake_label)
            discr_loss = real_loss + fake_loss
            discr_loss.backward()
            discr_optimizer.step()

            # Train Generator
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size, 100).to(device)
            fake_img = generator.forward(noise)
            fake_out = discriminator.forward(fake_img)
            gen_loss = lossFunc(fake_out, real_label)
            gen_loss.backward()
            gen_optimizer.step()
            print(f"Epoch: {i}, Batch: {j}, Discriminator Loss: {discr_loss.item()}, Generator Loss: {gen_loss.item()}")
            if j % 100 == 0:
                torchvision.utils.save_image(fake_img, f"output/{i}_{j}.png", nrow=10)









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    main(args.epoch, args.batch_size, args.lr)