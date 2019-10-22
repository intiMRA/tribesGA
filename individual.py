import numpy as np
class Individual:
    def __init__(self, genes=None, isRessessive=None, parentGenes=None):
        if parentGenes is None:
            parentGenes = []
        if isRessessive is None:
            isRessessive = []
        if genes is None:
            genes = []
        self.genes=genes
        self.isRessessive=isRessessive
        self.parentGenes=parentGenes
        self.fitness=-np.inf
        self.start=0
        self.end=0
