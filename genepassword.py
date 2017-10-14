import random
import datetime as time

def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)
#

def display(guess):
    timeDiff = time.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{0}\t{1}\t{2!s}".format(guess, fitness, timeDiff))
#

def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    #
    return "".join(genes)
#

def mutate(parent):
    index = random.randrange(0, len(parent))
    childGenes = list(parent)
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    return "".join(childGenes)
#

random.seed()
geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!. "
target = "Hello World!"
startTime = time.datetime.now()
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParent)

gen = 0

while True:
    gen += 1
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    
    if bestFitness >= childFitness:
        continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness
    bestParent = child
#

print("done in {0} generations".format(gen))