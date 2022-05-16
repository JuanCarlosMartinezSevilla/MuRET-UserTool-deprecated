class Messages:

    def sc():
        print("##########################################################")
        print("###  You are about to train a Symbol Classifier model  ###")
        print("##########################################################\n")

    def e2e():
        print("########################################################################")
        print("###  You are about to train an Staff-level symbol recognition model  ###")
        print("########################################################################\n")

    def new_images(images):
        print("###########################################")
        print("###  Launching Data Augmentation tool   ###")
        print("###########################################\n")
        print(f"\nFirst, we are going to create {images} new synthetic images. Please wait.\n")

    def using_document():
        print("##########################################################")
        print("###  You are about to train a Document Analysis model  ###")
        print("##########################################################\n")

    def welcome():
        print("=========================================")
        print("\nWelcome to MuRET's User Training Tool\n")
        print("=========================================\n")

    def end(args):
        print(f"========================================================================")
        print(f"\nThe tool has ended. You can find your results in {args.pkg_name}.tgz\n")
        print(f"========================================================================")

    def main():
        print("Messages main")

if __name__ == '__main__':
    Messages.main()