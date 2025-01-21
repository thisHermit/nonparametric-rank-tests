from .cat import Cat

class PersianCat(Cat):
    def __init__(self, name: str):
        super().__init__(name, sound="Soft Meow")

    def groom(self):
        print(f"{self.name} is being groomed. Persian cats love grooming!")
