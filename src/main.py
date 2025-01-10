import Generator
import Discriminator
def main():
    gen = Generator.Generator(64, 3)
    disc = Discriminator.Discriminator(64, 3)

if __name__ == "__main__":
    main()