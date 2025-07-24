import matplotlib.pyplot as plt
import numpy as np
from network import Network, load, save
from data_handling import get_data


class Ui:
    def __init__(self):
        self.net = Network([784, 32, 16, 10])
        self.training_data, self.validation_data, self.testing_data = get_data()
        initial_loss = self.net.validation_loss(self.training_data)
        initial_accuracy = self.net.validation_accuracy(self.validation_data)
        self.training_loss = [initial_loss]
        self.validation_accuracy = [initial_accuracy]

        self.instructions = "Instructions: \n"
        self.instructions += "0 = Quit \n"
        self.instructions += "1 = Load saved neural network \n"
        self.instructions += "2 = Create new neural network \n"
        self.instructions += "3 = Save neural network \n"
        self.instructions += "4 = Train neural network \n"
        self.instructions += "5 = Test neural network"

    def start(self):

        print(self.instructions)

        while True:
            print("Main selection:")
            action = self.action(["0", "1", "2", "3", "4", "5"], self.instructions)

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
            elif action == 5:
                self.test()

    def load_saved(self):
        try:
            self.net = load()
            print("Network loaded. \n")
        except FileNotFoundError:
            print("No neural network saved. Continuing with new. \n")

    def create_new(self):
        print("Create new neural network?")
        instr = "0 = cancel \n"
        instr += "1 = proceed"

        print(instr)

        action = self.action(["0", "1"], instr)

        if action == 0:
            return

        n_max = 10
        print(f"Give hidden layer sizes (max {n_max} layers) (Return blank when ready)")
        layers = [784]
        for _ in range(n_max):
            userinput = self.get_integer_input(1, 784, skip=True)
            if userinput == "":
                break

            layers.append(userinput)
            print(f"{userinput} appended to layers")

        layers.append(10)
        self.net = Network(layers)
        print(f"New network with layers {layers} created.")

        initial_loss = self.net.validation_loss(self.training_data)
        initial_accuracy = self.net.validation_accuracy(self.validation_data)
        self.training_loss = [initial_loss]
        self.validation_accuracy = [initial_accuracy]

    def save_net(self):
        save(self.net)
        print("Network saved.")

    def train(self):
        instructions = "Choose gradient descent algorithm: (0 = Cancel) \n"
        instructions += "1 = Vanilla \n"
        instructions += "2 = Stochastic \n"
        instructions += "3 = Minibatch"
        print(instructions)

        action = self.action(["0", "1", "2", "3"], instructions)
        if action == 0:
            return

        print("Enter amount of epochs:")
        epochs = self.get_integer_input(1, 500)

        print("Enter learning rate:")
        lr = self.get_float_input(0.00001, 99)

        if action == 1:
            print("Training...")
            training_loss, validation_accuracy = self.net.vanilla_gradient_descent(
                self.training_data, epochs, lr, self.validation_data)
        elif action == 2:
            print("Training...")
            training_loss, validation_accuracy = self.net.stochastic_gradient_descent(
                self.training_data, epochs, lr, self.validation_data)
        else:
            print("Enter minibatch size:")
            minibatch_size = self.get_integer_input(1, len(self.training_data))

            print("Training...")
            training_loss, validation_accuracy = self.net.minibatch_gradient_descent(
                self.training_data, minibatch_size, epochs, lr, self.validation_data)

        self.training_loss += training_loss
        self.validation_accuracy += validation_accuracy

        print("Training completed \n")

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Training loss", color="navy")
        ax1.set_ylim(0, max(self.training_loss))
        ax1.plot(self.training_loss, color="navy")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation accuracy", color="orangered")
        ax2.set_ylim(0, 1)
        ax2.plot(self.validation_accuracy, color="orangered")

        fig.tight_layout()
        ax1.grid()
        plt.show()

    def test(self):
        correct, incorrect = self.net.test_classification(self.testing_data)

        n = len(self.testing_data)
        n_corr = len(correct)
        n_incorr = len(incorrect)
        print(f"The neural network classified {n_corr} images correctly.")
        print(f"The neural network classified {n_incorr} images incorrectly.")
        print(f"Accuracy: {n_corr / n * 100:.4f}%")
        print()

        i_corr = 0
        i_incorr = 0
        while True:
            instr = "View images the network classified correctly/incorrectly: \n"
            instr += "0 = stop \n"
            instr += "1 = Correctly \n"
            instr += "2 = Incorrectly"

            print(instr)

            action = self.action(["0", "1", "2"], instr)

            if action == 0:
                break

            if action == 1:
                img, label = correct[i_corr]
                i_corr += 1
                title = f"{i_corr}/{n_corr}, {label}"
            else:
                img, label, classification = incorrect[i_incorr]
                i_incorr += 1
                cl = np.argmax(classification)
                title = f"{i_incorr}/{n_incorr}, "
                title += f"{label} that the network classified as {cl}"

            img = np.reshape(img, (28, 28))
            plt.title(title)
            plt.axis("off")
            plt.imshow(img, cmap="grey_r")
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
