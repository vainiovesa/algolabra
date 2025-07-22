from network import Network, load, save
from data_handling import get_data

class Ui:
    def __init__(self):
        self.net = Network([784, 32, 16, 10])
        self.training_data, self.validation_data, self.testing_data = get_data()

    def start(self):

        self.instructions()

        while True:
            action = self.action()

            if action == 0:
                break

            if action == 1:
                self.load_saved()
            elif action == 2:
                self.create_new()
            elif action == 3:
                self.save_net()

    def load_saved(self):
        try:
            self.net = load()
            print("Network loaded.")
        except FileNotFoundError:
            print("No neural network saved. Continuing with new.")

    def create_new(self):
        self.net = Network([784, 32, 16, 10])
        print("New network created.")

    def save_net(self):
        save(self.net)
        print("Network saved.")

    def action(self):
        action = input("> ")
        while action not in ("0", "1", "2", "3"):
            self.instructions()
            action = input("> ")
        return int(action)

    def instructions(self):
        print()
        print("Instructions:")
        print("0 = Quit")
        print("1 = Load saved neural network")
        print("2 = Create new neural network")
        print("3 = Save neural network")
        print()

if __name__=="__main__":
    ui = Ui()
    ui.start()
