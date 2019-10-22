import numpy as np


class Individual:
    def __init__(self, genes=None, isRessessive=None, parentGenes=None):
        if parentGenes is None:
            parentGenes = []
        if isRessessive is None:
            isRessessive = []
        if genes is None:
            genes = []
        self.genes = genes
        self.isRessessive = isRessessive
        self.parentGenes = parentGenes
        self.fitness = -np.inf
        self.start = 0
        self.end = 0

    def copy(self):
        ind = Individual(genes=self.genes[:], isRessessive=self.isRessessive[:], parentGenes=self.parentGenes[:])
        ind.fitness = self.fitness
        ind.start = self.start
        ind.end = self.end
        return ind
