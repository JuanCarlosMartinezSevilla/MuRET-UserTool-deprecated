from rich import print

class Messages:

    def sc():
        print("\nYou are about to train a [bold green]Symbol Classifier[/bold green] model")

    def e2e():
        print("\nYou are about to train an [bold green]End to End[/bold green] model")

    def new_images(images):
        print(f"\nFirst, we are going to create [bold]{images}[/bold] new synthetic images. Please wait.\n")

    def using_document():
        print("\nYou are about to train a [bold green]Document Analysis[/bold green] model")

    def welcome():
        print("\n:musical_notes: Welcome to [bold cyan u]MuRET User's tool[/bold cyan u] :musical_note:")

    def end():
        print("\n[yellow]The tool has ended. You can find your results in [bold]MuRETPackage.tgz[/bold][yellow]\n")

    def main():
        print("Messages main")

if __name__ == '__main__':
    Messages.main()