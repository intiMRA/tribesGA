import numpy as np

filePath = '/vol/grid-solar/sgeusers/resendinti/'


class Dataset:
    def __init__(self, file=None, x=None, y=None, cathegorical=False, removeCathegoricalX=False):
        if file:
            x = []
            y = []
            for row in file:

                if len(row)>1:
                    x.append(row[:len(row) - 1])
                    y.append(row[-1])

        if cathegorical:
            y = self.toNumerical(y)
        if removeCathegoricalX:
            for i in range(len(x)):
                X=[]
                for j in range(len(x[0])):
                    try:
                        float(x[i][j])
                        X.append(x[i][j])
                    except:
                        pass

                x[i]=self.toNumerical(x[i])
        self.data = np.array(x).astype(np.float64)
        self.target = np.array(y).astype(np.float64)

    def toNumerical(self, y):
        Y = []
        i = 0
        classes = {}
        for inst in y:
            try:
                float(inst)
                if inst not in classes:
                    classes[inst] = inst
            except:
                if inst not in classes:
                    classes[inst] = i
                    i += 1
        for inst in y:
            Y.append(classes[inst])
        return Y


def getSeeds():
    return [int(s) for s in open(filePath + "seeds.txt")]


def getAd():
    test=[f.split(",") for f in open(filePath + "DATASETS/ad data/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/ad data/trainingFile.txt")]
    return {"test":Dataset(file=test),"train":Dataset(file=train),"name":"Ad"}

def getTest():
    test=[f.split(",") for f in open(filePath + "DATASETS/semeinon/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/semeinon/trainingFile.txt")]
    return {"test":Dataset(file=test),"train":Dataset(file=train),"name":"Test"}

def getPop():
    test=[f.split(",") for f in open(filePath + "DATASETS/popularity/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/popularity/trainingFile.txt")]
    return {"test":Dataset(file=test),"train":Dataset(file=train),"name":"Popularity"}


def getHuman():
    test=[f.split(",") for f in open(filePath + "DATASETS/human/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/human/trainingFile.txt")]
    return {"test":Dataset(file=test),"train":Dataset(file=train),"name":"Human"}

def getMobile():
    test=[f.split(",") for f in open(filePath + "DATASETS/mobile/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/mobile/trainingFile.txt")]
    return {"test":Dataset(file=test),"train":Dataset(file=train),"name":"Mobile"}


def getAry():
    test=[f.split(",") for f in open(filePath + "DATASETS/arrhythmia/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/arrhythmia/trainingFile.txt")]
    return {"train":Dataset(file=train),"test":Dataset(file=test), "name":"Ary"}


def getmfeat_216():
    test = [f.split(",") for f in open(filePath + "DATASETS/number features/testFile216.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/number features/trainingFile216.txt")]
    return {"train":Dataset(file=train),"test":Dataset(file=test), "name":"Mfeat_216"}



def getmfeat_240():
    test = [f.split(",") for f in open(filePath + "DATASETS/number features/testFile240.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/number features/trainingFile240.txt")]
    return {"train":Dataset(file=train),"test":Dataset(file=test), "name":"Mfeat_240"}


def getUCI():
    test = [f.split(",") for f in open(filePath + "DATASETS/UCI HAR Dataset/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/UCI HAR Dataset/trainingFile.txt")]
    return {"train":Dataset(file=train),"test":Dataset(file=test), "name":"UIC"}



def getTruck():
    test = [f.split(",") for f in open(filePath + "DATASETS/aps/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/aps/trainingFile.txt")]
    return {"train":Dataset(file=train,cathegorical=True),"test":Dataset(file=test,cathegorical=True), "name":"Truck"}


def getLibras():
    test = [f.strip("\n").split(",") for f in open(filePath + "DATASETS/libras/testFile.txt")]
    train = [f.strip("\n").split(",") for f in open(filePath + "DATASETS/libras/trainingFile.txt")]
    return {"train":Dataset(file=train),"test":Dataset(file=test), "name":"Libras"}


def getML():
    test = [f.split(",") for f in open(filePath + "DATASETS/ml prove/testFile.txt")]
    train = [f.split(",") for f in open(filePath + "DATASETS/ml prove/trainingFile.txt")]
    return {"train":Dataset(file=train),"test":Dataset(file=test), "name":"ML"}


def getAll():
    return getAry(), getMobile(), getAd(), getUCI(), getLibras(), getHuman(), getmfeat_216(), getmfeat_240(), getML(), getPop()


def getDataset(number):
    if number == 0:
        return getAry()
    elif number == 1:
        return getMobile()
    elif number == 2:
        return getAd()
    elif number == 3:
        return getUCI()
    elif number == 4:
        return getLibras()
    elif number == 5:
        return getHuman()
    elif number == 6:
        return getmfeat_216()
    elif number == 7:
        return getmfeat_240()
    elif number == 8:
        return getML()
    else:
        return getPop()

def printSizes():
    ds=getAll()

    for d in ds:
        print("name",d["name"],"inst",len(d["test"].data)+len(d["train"].data),"feat",len(d["test"].data[0]))




