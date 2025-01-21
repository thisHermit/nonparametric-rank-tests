from animal_package import Animal, Dog
from animal_package.cat import Cat, PersianCat

# Create instances
generic_animal = Animal("Generic Animal", "Some sound")
cat1 = Cat("Whiskers")
cat2 = PersianCat("Snowball")
dog = Dog("Buddy")

# General animal-to-animal communication
generic_animal.call_other_animal(cat1)
cat1.call_other_animal(dog)
dog.call_other_animal(cat2)

# Cat-specific communication
cat1.call_another_cat(cat2)
cat1.call_another_cat(dog)

# Test specific behaviors
cat1.climb()
dog.fetch()
cat2.groom()
