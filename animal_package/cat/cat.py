from animal_package.animal import Animal

class Cat(Animal):
    def __init__(self, name: str, sound: str = "Meow"):
        super().__init__(name, sound)

    def climb(self):
        print(f"{self.name} is climbing a tree.")

    def call_another_cat(self, other_cat):
        if isinstance(other_cat, Cat):
            print(f"{self.name} is calling {other_cat.name}: {self.sound}!")
            other_cat.respond_to_call(self)
        else:
            print(f"{self.name} can only call another cat, not a {type(other_cat).__name__}.")
