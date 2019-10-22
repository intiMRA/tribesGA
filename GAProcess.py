from individual import Individual
import random as rd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import Data
from sklearn.naive_bayes import GaussianNB
def createIndividual(possibleGenes=None, sizeIndividual=None,ressessiveChance=0.5,start=0,end=0,defaultGene=None)->Individual:
    if possibleGenes is None:
        possibleGenes = [0, 1]
    if sizeIndividual==None:
        return Individual()
    genes = []
    for index in range(sizeIndividual):
        if index>=start and index<=end:
            genes.append(rd.choice(possibleGenes))
        else:
            genes.append(defaultGene)

    ressesive=[rd.uniform(0,1)<ressessiveChance for i in range(sizeIndividual)]
    ind=Individual(genes=genes,isRessessive=ressesive)
    ind.start=start
    ind.end=end
    return ind

def createTribes(possibleGenes=None, individualSize=None, numTribes=1, defaultGene=None, popSize=100,ressessiveChance=0.5,numberInAtribe=None)->list:
    tribes=[]
    if possibleGenes is None:
        possibleGenes = [0, 1]
    if numberInAtribe==None:
        numberInAtribe=popSize

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

def cross(individual1:Individual,individual2:Individual,newressessiveChance=0.3,defaultGene=None)->Individual:
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
            if individual2.genes[i]==defaultGene and individual1.genes[i]==defaultGene:
                parentGenes.append(defaultGene)
                continue

            if individual1.genes[i]==defaultGene:
                parentGenes.append(defaultGene)
                genes[i]=individual2.genes[i]
                ressesive[i]=individual2.isRessessive[i]
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

                gene,ress = getGenes(pg1,pg2,individual1.genes[i],individual2.genes[i],defaultGene)
                isRes = True
            genes[i]=gene
            ressesive[i]=isRes
            parentGenes.append(ress)
    ind=Individual(genes=genes,isRessessive=ressesive,parentGenes=parentGenes)
    ind.start=min(individual1.start,individual2.start)
    ind.end=max(individual1.end,individual2.end)
    return ind

def getGenes(pg1,pg2,g1,g2,defaultGene):
    if pg1==defaultGene and pg2==defaultGene:
        gene = rd.choice([g1, g2])
        if gene == g1:
            ress = g2
        else:
            ress = g1
        return gene,ress

    if pg1 == defaultGene:
        ar=[g1, g2,pg2]
        gene = rd.choice(ar)
        ar.remove(gene)
        ress=rd.choice(ar)
        return gene, ress

    if pg2 == defaultGene:
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
                individual.isRessessive[i]=rd.uniform(0,1)<ressessiveChance
    return individual

def fitness(individual,X,Y,classifier=KNeighborsClassifier(n_jobs=-1,n_neighbors=3),defaultGene=None):
    fit=0
    kf=StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        TX=[]
        for x in X_train:
            Fx = []
            for i in range(len(individual.genes)):
                if individual.genes[i]==1:
                    Fx.append(x[i])
            TX.append(Fx)
        TEX = []
        for x in X_test:
            Fx = []
            for i in range(len(individual.genes)):
                if individual.genes[i]==1:
                    Fx.append(x[i])
            TEX.append(Fx)

        try:
            model=classifier.fit(np.array(TX),y_train)
            pre=model.predict(np.array(TEX))
            n=0
            for tv,p in zip(y_test,pre):
                if tv==p:
                    n+=1
            n/=len(y_test)
            fit+=n
        except Exception as e:
            return -np.inf

    return fit/10


def evaluate(individual,X,Y,testX,testY,classifier=KNeighborsClassifier(n_jobs=-1,n_neighbors=3),defaultGene=None):
    TX=[]
    for x in X:
        Fx = []
        for i in range(len(individual.genes)):
            if individual.genes[i]==1:
                Fx.append(x[i])
        TX.append(Fx)
    TEX = []
    for x in testX:
        Fx = []
        for i in range(len(individual.genes)):
            if individual.genes[i]==1:
                Fx.append(x[i])
        TEX.append(Fx)

    model=classifier.fit(np.array(TX),Y)
    pre=model.predict(np.array(TEX))
    n=0
    for tv,p in zip(testY,pre):
        if tv==p:
            n+=1
    n/=len(testY)
    return n


def selection(pop, popsize, k,possibleGenes=None,ressessiveChance=0.5,defaultGene=None):
    newPop=[]
    best=pop[0] #type: Individual
    for ind in pop:
        ind=ind #type: Individual
        if ind.fitness> best.fitness:
            best=ind
    for i in range(popsize-1):

        l=rd.choices(pop,k=k)
        ind=l[0]
        for c in l:
            if c.fitness>ind.fitness:
                ind=c
        newPop.append(ind)
    inPop=False
    for ind in newPop:
        if np.array_equal(ind.genes,best.genes) and np.array_equal(ind.isRessessive,best.isRessessive):
            inPop=True
            break
    if not inPop:
        newPop.append(best)
    else:
        newPop.append(createIndividual(possibleGenes,len(best.genes),ressessiveChance,best.start,best.end,defaultGene))
    return newPop

def injectNew(individual,tribe):
    index=0
    for i,ind in enumerate(tribe):
        if tribe[index].fitness<ind.fitness:
            index=i
    tribe.pop(index)
    tribe.append(individual.copy())
    return tribe

def globalTribe(tribes,popSize):
    gtribe=[]
    n=max(popSize//len(tribes),1)
    for t in tribes:
        st=sorted(t,key=lambda x:x.fitness,reverse=True)
        for i in range(0,n):
            gtribe.append(st[i])
    return gtribe

def EAProcesses(n,popsize, mutProb, migrationProb, crossProb, possibleGenes=None, individualSize=100, numTribes=1, defaultGene=None, numberInAtribe=None, k=3, ressessiveChance=0.5,X=None,Y=None,classifier=KNeighborsClassifier(n_jobs=-1,n_neighbors=3),mergeHalfWay=False)->Individual:
    psize=max(popsize,numTribes)
    if possibleGenes is None:
        possibleGenes = [0, 1]
    tribes=createTribes(possibleGenes,individualSize,numTribes,defaultGene,psize,ressessiveChance,numberInAtribe)
    for t in tribes:
        for i in t:
            i.fitness=fitness(i,X,Y,classifier=classifier,defaultGene=defaultGene)
    gTribe=None
    for gen in range(n):
        print("iteration",gen+1,"out of",n)
        if(gen<(n//2) or not mergeHalfWay):
            for i in range(len(tribes)):
                maxInd = tribes[i][0]
                minInd = tribes[i][-1]
                invalid = 0
                for index in range(len(tribes[i])):
                    if maxInd.fitness < tribes[i][index].fitness:
                        maxInd = tribes[i][index]
                    if minInd.fitness > tribes[i][index].fitness:
                        minInd = tribes[i][index]
                    if tribes[i][index].fitness==-np.inf:
                        invalid+=1
                    if rd.uniform(0,1)<crossProb:
                        tribes[i][index]=cross(tribes[i][index],rd.choice(tribes[i]))

                for index in range(len(tribes[i])):
                    if rd.uniform(0,1)<mutProb:
                        tribes[i][index]=mutate(tribes[i][index],ressessiveChance,possibleGenes=possibleGenes)

                for index in range(len(tribes[i])):
                    if rd.uniform(0,1)<migrationProb:
                        injectNew(tribes[i][index],rd.choice(tribes))


                print("tribe:",i,"best:",maxInd.fitness,"worst:",minInd.fitness,"invalid:",invalid)
                for index in range(len(tribes[i])):
                    if tribes[i][index].fitness==-np.inf:
                        tribes[i][index].fitness=fitness(tribes[i][index],X,Y,classifier,defaultGene)
                tribes[i]=selection(tribes[i],psize,k,possibleGenes,ressessiveChance,defaultGene)
            continue
        elif mergeHalfWay:
            if  gen==(n//2) or not gTribe:
                gTribe=globalTribe(tribes,popsize)
            invalid=0
            for i in gTribe:
                if i.fitness==-np.inf:
                    invalid+=1
            maxInd=gTribe[0]
            minInd=gTribe[-1]
            for index in range(len(gTribe)):

                if maxInd.fitness < gTribe[index].fitness:
                    maxInd = gTribe[index]
                if minInd.fitness > gTribe[index].fitness:
                    minInd = gTribe[index]
                if rd.uniform(0, 1) < crossProb:
                    gTribe[index] = cross(gTribe[index], rd.choice(gTribe))

            for index in range(len(gTribe)):
                if rd.uniform(0, 1) < mutProb:
                    gTribe[index] = mutate(gTribe[index], ressessiveChance, possibleGenes=possibleGenes)
            print("tribe:", "global", "best:", maxInd.fitness, "worst:", minInd.fitness,"invalid",invalid)
            for index in range(len(gTribe)):
                if gTribe[index].fitness == -np.inf:
                    gTribe[index].fitness = fitness(gTribe[index], X, Y, classifier,defaultGene)
            gTribe = selection(gTribe, popsize, k, possibleGenes, ressessiveChance, defaultGene)
    if not mergeHalfWay:
        gTribe = globalTribe(tribes, popsize)

    return sorted(gTribe,key=lambda x:x.fitness,reverse=True)[0]


def main():
    dataset=Data.getLibras()
    trainingX = dataset["train"].data
    trainingY = dataset["train"].target
    testX = dataset["test"].data
    testY = dataset["test"].target
    ind=EAProcesses(n=50,popsize=100,mutProb=0.3,migrationProb=0.01,crossProb=0.7,individualSize=len(testX[0]),numTribes=10,X=trainingX,Y=trainingY,mergeHalfWay=False)
    print("final score on test:",evaluate(ind,trainingX,trainingY,testX,testY),"final score on trining:",ind.fitness)

main()







