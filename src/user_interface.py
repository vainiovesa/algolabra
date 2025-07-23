import matplotlib.pyplot as plt
from network import Network, load, save
from data_handling import get_data


class Ui:
    def __init__(self):
        self.net = Network([784, 32, 16, 10])
        self.training_data, self.validation_data, self.testing_data = get_data()

        self.instructions = "Instructions: \n"
        self.instructions += "0 = Quit \n"
        self.instructions += "1 = Load saved neural network \n"
        self.instructions += "2 = Create new neural network \n"
        self.instructions += "3 = Save neural network \n"
        self.instructions += "4 = Train neural network"

    def start(self):

        print(self.instructions)

        while True:
            print("Main selection:")
            action = self.action(["0", "1", "2", "3", "4"], self.instructions)

            if action == 0:
                break

            if action == 1:
                self.load_saved()
            elif action == 2:
                self.create_new()
            elif action == 3:
                self.save_net()
            elif action == 4:
                self.train()

    def load_saved(self):
        try:
            self.net = load()
            print("Network loaded. \n")
        except FileNotFoundError:
            print("No neural network saved. Continuing with new. \n")

    def create_new(self):
        self.net = Network([784, 32, 16, 10])
        print("New network created.")

    def save_net(self):
        save(self.net)
        print("Network saved.")

    def train(self):
        instructions = "Choose gradient descent algorithm: \n"
        instructions += "1 = Vanilla \n"
        instructions += "2 = Stochastic \n"
        instructions += "3 = Minibatch"
        print(instructions)

        action = self.action(["1", "2", "3"], instructions)

        print("Enter amount of epochs:")
        epochs = self.get_integer_input(2, 500)

        print("Enter learning rate:")
        lr = self.get_float_input(0.00001, 99)

        if action == 1:
            print("Training...")
            training_loss, validation_loss = self.net.vanilla_gradient_descent(
                self.training_data, epochs, lr, self.validation_data)
        elif action == 2:
            print("Training...")
            training_loss, validation_loss = self.net.stochastic_gradient_descent(
                self.training_data, epochs, lr, self.validation_data)
        else:
            print("Enter minibatch size:")
            minibatch_size = self.get_integer_input(1, len(self.training_data))

            print("Training...")
            training_loss, validation_loss = self.net.minibatch_gradient_descent(
                self.training_data, minibatch_size, epochs, lr, self.validation_data)

        print("Training completed \n")
        plt.plot(training_loss, label="Training loss")
        plt.plot(validation_loss, label="Validation loss")
        plt.xticks(range(epochs), range(1, epochs + 1))
        plt.legend()
        plt.grid()
        plt.show()

    def action(self, choices: list, instructions: str):
        action = input("\n> ")
        print()
        while action not in choices:
            print(instructions)
            action = input("\n> ")
            print()
        return int(action)

    def get_integer_input(self, a: int, b: int, skip: bool = False):
        userinput = input("\n> ")
        print()
        while True:
            try:
                userinput = int(userinput)
                if a <= userinput <= b:
                    break

                print(f"The integer must be between {a} and {b}.")
                userinput = input("\n> ")
                print()
            except ValueError:
                if skip and userinput == "":
                    break

                print("Enter an integer.")
                userinput = input("\n> ")
                print()
        return userinput

    def get_float_input(self, a: float, b: float, skip: bool = False):
        userinput = input("\n> ")
        print()
        while True:
            try:
                userinput = float(userinput)
                if a <= userinput <= b:
                    break

                print(f"The value must be between {a} and {b}.")
                userinput = input("\n> ")
                print()
            except ValueError:
                if skip and userinput == "":
                    break

                print("Enter a float value.")
                userinput = input("\n> ")
                print()
        return userinput
