class Animal:
    def __init__(self, name: str, sound: str):
        self.name = name
        self.sound = sound

    def make_sound(self):
        print(f"{self.name} makes a sound: {self.sound}")

    def call_other_animal(self, other_animal):
        print(f"{self.name} is calling {other_animal.name}: {self.sound}!")
        other_animal.respond_to_call(self)

    def respond_to_call(self, caller):
        print(f"{self.name} responds to {caller.name}: {self.sound}!")
