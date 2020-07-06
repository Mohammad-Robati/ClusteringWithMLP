class Data:

    def __init__(self, file):
        self.trainingSet = []
        self.testSet = []
        self.getInfo(file)

    def getInfo(self, file):
        f = open(file, 'r')
        f.readline()
        flag = True
        for line in f:
            x, y, d = line.split(",")
            if flag:
                self.testSet.append([float(x), float(y), int(d[0])])
            else:
                self.trainingSet.append([float(x), float(y), int(d[0])])
            flag = not flag
