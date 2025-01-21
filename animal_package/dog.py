from .animal import Animal

class Dog(Animal):
    def __init__(self, name: str, sound: str = "Bark"):
        super().__init__(name, sound)

    def fetch(self):
        print(f"{self.name} is fetching the ball.")
