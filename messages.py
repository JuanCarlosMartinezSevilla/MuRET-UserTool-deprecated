class Messages:

    def sc():
        print("\nYou are about to train a Symbol Classifier model")

    def e2e():
        print("\nYou are about to train an End to End model")

    def new_images(images):
        print(f"\nFirst, we are going to create {images} new synthetic images. Please wait.\n")

    def using_document():
        print("\nYou are about to train a Document Analysis model")

    def welcome():
        print("\nWelcome to MuRET User's tool")

    def end(args):
        print(f"\nThe tool has ended. You can find your results in {args.pkg_name}.tgz\n")

    def main():
        print("Messages main")

if __name__ == '__main__':
    Messages.main()