class Messages:

    def sc():
        print("\n##########################################################")
        print("#    You are about to train a Symbol Classifier model    #")
        print("##########################################################\n")

    def e2e():
        print("\n########################################################################")
        print("#    You are about to train an Staff-level symbol recognition model    #")
        print("########################################################################\n")

    def new_images(images):
        print("\n###########################################")
        print("#    Launching Data Augmentation tool     #")
        print("###########################################\n")
        print(f"\nFirst, we are going to create {images} new synthetic images. Please wait.\n")

    def using_document():
        print("\n##########################################################")
        print("#    You are about to train a Document Analysis model    #")
        print("##########################################################\n")

    def welcome():
        print("=========================================")
        print("\n  Welcome to MuRET's User Training Tool\n")
        print("=========================================\n")

    def end(args):
        print(f"===============================================================================")
        print(f"\n  The tool has ended. You can find your trained models in {args.pkg_name}.tgz\n")
        print(f"===============================================================================\n")

    def main():
        print("Messages main")

if __name__ == '__main__':
    Messages.main()