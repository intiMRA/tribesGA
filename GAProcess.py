from individual import Individual
import random as rd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
def createIndividual(possibleGenes=None, sizeIndividual=None,ressessiveChance=0.5,start=0,end=0,defaultGene=None)->Individual:
    if possibleGenes is None:
        possibleGenes = [0, 1]
    if sizeIndividual==None:
        return Individual()
    genes = []
    for index in range(sizeIndividual):
        if index>start and index<end:
            genes.append(rd.choice(possibleGenes))
        else:
            genes.append(defaultGene)

    ressesive=[lambda x: rd.uniform(0,1)<ressessiveChance for i in range(sizeIndividual)]
    ind=Individual(genes=genes,isRessessive=ressesive)
    ind.start=start
    ind.end=end
    return ind

def createTribes(possibleGenes=None, individualSize=None, numTribes=1, defaultGene=None, popSize=100,ressessiveChance=0.5)->list:
    tribes=[]
    if possibleGenes is None:
        possibleGenes = [0, 1]
    numberInAtribe = popSize // numTribes

    if individualSize==None:
        return
    sizeTopopulate= individualSize // numTribes
    for i in range(numTribes):
        start=i*sizeTopopulate
        end=(i+1)*sizeTopopulate
        tribe=[]
        for t in range(numberInAtribe):
            tribe.append(createIndividual(possibleGenes,individualSize,ressessiveChance,start,end,defaultGene))
        tribes.append(tribe)
    return tribes

def cross(individual1:Individual,individual2:Individual,newressessiveChance=0.3)->Individual:
    genes=individual1.genes[:]
    ressesive=individual1.isRessessive[:]
    parentGenes=[]

    if (individual1.start>individual2.start and individual1.end>individual2.end) or (individual2.start>individual1.start and individual2.end>individual1.end):
        for i in range(len(genes)):
            if i>=individual2.start and i<individual2.end:
                genes[i]=individual1.genes[i]
                ressesive[i]=individual2.isRessessive[i]
    else:
        for i in range(len(genes)):
            if individual2.genes[i]==None and individual1.genes[i]==None:
                parentGenes.append(None)
                continue

            if individual1.genes[i]==None:
                parentGenes.append(None)
                genes[i]=individual2.genes[i]
                ressesive=individual2.isRessessive[i]
                continue

            if individual1.isRessessive[i] and not individual2.isRessessive[i]:
                gene=individual2.genes[i]
                ress=individual1.genes[i]
                isRes=rd.uniform(0,1)<newressessiveChance
            elif individual2.isRessessive[i] and not individual1.isRessessive[i]:
                gene=individual1.genes[i]
                ress=individual2.genes[i]
                isRes=rd.uniform(0,1)<newressessiveChance
            elif not individual1.isRessessive[i]:
                gene=rd.choice([individual1.genes[i],individual2.genes[i]])
                if gene==individual1.genes[i]:
                    ress=individual2.genes[i]
                else:
                    ress=individual1.genes[i]
                isRes=False
            else:
                pg1=None
                pg2=None
                if len(individual1.parentGenes)>1:
                    pg1=individual1.parentGenes[i]

                if len(individual2.parentGenes)>1:
                    pg2=individual2.parentGenes[i]

                gene,ress = getGenes(pg1,pg2,individual1.genes[i],individual2.genes[i])
                isRes = True
            genes[i]=gene
            ressesive[i]=isRes
            parentGenes.append(ress)

    ind=Individual(genes=genes,isRessessive=ressesive,parentGenes=parentGenes)
    ind.start=min(individual1.start,individual2.start)
    ind.end=max(individual1.end,individual2.end)
    return ind

def getGenes(pg1,pg2,g1,g2):
    if pg1==None and pg2==None:

        gene = rd.choice([g1, g2])
        if gene == g1:
            ress = g2
        else:
            ress = g1
        return gene,ress

    if pg1 == None:
        ar=[g1, g2,pg2]
        gene = rd.choice(ar)
        ar.remove(gene)
        ress=rd.choice(ar)
        return gene, ress

    if pg2 == None:
        ar=[g1, g2,pg1]
        gene = rd.choice(ar)
        ar.remove(gene)
        ress=rd.choice(ar)
        return gene, ress

    ar = [g1, g2, pg1,pg2]
    gene = rd.choice(ar)
    ar.remove(gene)
    ress = rd.choice(ar)
    return gene, ress

def mutate(individual:Individual, ressessiveChance=0.5, rate=0.05, possibleGenes=None)->Individual:
    if possibleGenes is None:
        possibleGenes = [0, 1]
    for i in range(len(individual.genes)):
        if i>=individual.start and i<individual.end:
            if rd.uniform(0,1)<rate:
                individual.genes[i]=rd.choice(possibleGenes)
                individual.isRessessive=rd.uniform(0,1)<ressessiveChance
    return individual

def fitness(X,Y,classifier=KNeighborsClassifier(n_jobs=-1,n_neighbors=3)):
    fit=0
    kf=StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        try:
            model=classifier.fit(X_train,y_train)
            pre=model.predict(X_test)
            n=0
            for tv,p in zip(y_test,pre):
                if tv==p:
                    n+=1
            n/=len(y_test)
            fit+=n
        except:
            return -np.inf

    return fit/10
