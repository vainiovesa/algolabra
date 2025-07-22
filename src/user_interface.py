class Ui:
    def __init__(self):
        pass

    def start(self):

        self.instructions()

        while True:
            action = self.action()

            if action == 0:
                break

    def action(self):
        action = input("> ")
        while action not in ("0", ):
            self.instructions()
            action = input("> ")
        return int(action)

    def instructions(self):
        print()
        print("Instructions:")
        print("0 = quit")
        print()
