import unittest
import datetime
import genetic
import random
import sys


class Resource:
    def __init__(self, name, value, weight, volume):
        self.name = name
        self.value = value
        self.weight = weight
        self.volume = volume


class Fitness:
    def __init__(self, totalWeight, totalVolume, totalValue):
        self.totalWeight = totalWeight
        self.totalVolume = totalVolume
        self.totalValue = totalValue

    def __gt__(self, other):
        return self.totalValue > other.totalValue

    def __str__(self):
        return "wt: {:0.2f} vol: {:0.2f} value: {}".format(
            self.totalWeight, self.totalVolume, self.totalValue)


class ItemQuantity:
    def __init__(self, item, quantity):
        self.item = item
        self.quantity = quantity

    def __eq__(self, other):
        return self.item == other.item and self.quantity == other.quantity


class KnapsackTests(unittest.TestCase):
    def test_cookies(self):
        items = [
            Resource("Flour", 1680, 0.265, 0.41),
            Resource("Butter", 1440, 0.5, 0.13),
            Resource("Sugar", 1840, 0.441, 0.29)
        ]
        maxWeight = 10
        maxVolume = 4


def get_fitness(genes):
    totalWeight = 0
    totalVolume = 0
    totalValue = 0

    for iq in genes:
        count = iq.quantity
        totalWeight += iq.item.weight * count
        totalVolume += iq.item.volume * count
        totalValue += iq.item.value * count

    return Fitness(totalWeight, totalVolume, totalValue)


def mutate(genes, items, maxWeight, maxVolume):
    fitness = get_fitness(genes)
    remainingWeight = maxWeight - fitness.totalWeight
    remainingVolume = maxVolume - fitness.totalVolume


def create(items, maxWeight, maxVolume):
    genes = []
    remainingWeight, remainingVolume = maxWeight, maxVolume
    for i in range(random.randrange(1, len(items))):
        newGene = add(genes, items, remainingWeight, remainingVolume)
        if newGene is not None:
            genes.append(newGene)
            remainingWeight -= newGene.quantity * newGene.item.weight
            remainingVolume -= newGene.quantity * newGene.item.volume
    return genes


def add(genes, items, maxWeight, maxVolume):
    usedItems = {iq.item for iq in genes}
    item = random.choice(items)
    while item in usedItems:
        item = random.choice(items)

    maxQuantity = max_quantity(item, maxWeight, maxVolume)
    return ItemQuantity(item, maxQuantity) if maxQuantity > 0 else None


def max_quantitiy(item, maxWeight, maxVolume):
    return min(
        int(maxWeight / item.weight) if item.weight > 0 else sys.maxsize,
        int(maxVolume / item.volume) if item.volume > 0 else sys.maxsize)
