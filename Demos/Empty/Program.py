from ai4animation import AI4Animation


class Program:
    def __init__(self, variable):
        self.Variable = variable

    def Start(self):
        # Initialize things here
        print(self.Variable)

    def Update(self):
        # Update things here
        return

    def Draw(self):
        # Draw things here
        return

    def GUI(self):
        # GUI things here
        return


def main():
    ai4a = AI4Animation(Program("Hello World"), mode=AI4Animation.Mode.HEADLESS)


if __name__ == "__main__":
    main()
