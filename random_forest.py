import math


#this will be our desicion tree
class treeNode(object):
    def __init__(self, nodeType,value):
        self.nodeType = nodeType #we have endPoint, option and question
        self.value = value
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)






#getting the data
def getMatrix(name):
    arr = []
    f = open (name,"r")
    #read line into array
    for line in f.readlines():
        # add a new sublist
        arr.append([])
        # loop over the elemets, split by whitespace
        for i in line.split():
            # convert to integer and append to the last
            # element of the list
            arr[-1].append(int(i))
    f.close()
    return arr

def getXMatrix():
    Data = []
    Data =getMatrix('Xtrain')
    return Data

def getXYMatrix():
    Data = []
    xData =getMatrix('Xtrain')
    yData =getMatrix('Ytrain')
    Data = [xData[i] + yData [i] for i in range(len(xData))]
    return Data





#actions on a matrix
def removeColumn(matrix,column):
    m=zip(*matrix)
    del m[column]
    m=zip(*m)
    return m


def getRowsThatHasValueInColumn(matrix,column,value):
    arr = []
    for line in matrix[:]:
        if line[column]==value:
            arr.append(line)
    return arr

##column starts from 0
def getYForValueAtSpesificColumn(matrix,column,value):
    arr = []
    for line in matrix[:]:
        if line[column]==value:
            arr.append(line)
    arr=getLastColumn(arr)
    return arr

def getLastColumn(matrix):
    return getColumn(matrix,len(matrix[0])-1)

def getColumn(matrix,column):
     return [(x[column]) for x in matrix]









#some statistic calculations

##enter a list of integer here
def getAntropy (y):
    if len(y) == 0:
        print ('we have zero length vector here..')
        return 0
    show=1.0*sum(y)/len(y)
    if show == 0 or show ==1:
        return 0 #if all 1 or all 0, we have no news, the antropy is zero
    antropy=-show*math.log(show)/math.log(2.0)-(1-show)*math.log(1-show)/math.log(2.0)
    return antropy

##you have a matrix and you want to know for a specific value at a specific column what is the antropy of y for that value at that column
def getYAntropyForSpecificXValue(matrix,column,xValue):
    y=getYForValueAtSpesificColumn(matrix,column,xValue);
    antropy=getAntropy(y)
    return antropy


def getInformationGainForSpecificXColumn (xyMatrix,column, yAntropy):
    informationGain=yAntropy;
    total=len(getColumn(xyMatrix,column))
    for value in set(getColumn(xyMatrix,column)):
        valueShow=1.0*getColumn(xyMatrix,column).count(value)/total
        antropyForSpecificValue=getYAntropyForSpecificXValue(xyMatrix,column,value)
        informationGain-=valueShow*antropyForSpecificValue
    return informationGain


def findColumnWithBestInformationGain(xyMatrix,yAntropy):
    maxInfoGain=-1.0
    bestColumn=-1
    for column in range (len(xyMatrix[0])-1):#we don't want to check also the y column
        gain=getInformationGainForSpecificXColumn(xyMatrix,column,yAntropy)
        if(gain>maxInfoGain):
            maxInfoGain=gain
            bestColumn=column
    return bestColumn


def checkMatrix(xyMatrix,ColumnId,aNode):
    yAntropy=getAntropy(getLastColumn(xyMatrix))
    if (yAntropy == 0):
        print ('we got a desicion, the answer is ',getLastColumn(xyMatrix)[0],'go back one step up')
        newNode=treeNode('endPoint',getLastColumn(xyMatrix)[0])
        aNode.add_child(newNode)
    else:
        bestColumn=findColumnWithBestInformationGain(xyMatrix,yAntropy)
        bestColumnValues=getColumn(xyMatrix,bestColumn)
        bestColumnId=ColumnId [bestColumn]
        newNode=treeNode('question',bestColumnId)
        aNode.add_child(newNode)
        aNode=aNode.children[0]
        del (ColumnId [bestColumn])
        for optionsInBestColumn in set(bestColumnValues):
            subMatrix=getRowsThatHasValueInColumn(xyMatrix,bestColumn,optionsInBestColumn)
            subMatrix=removeColumn(subMatrix,bestColumn)
            print ('go to question ',bestColumnId, 'for answer ',optionsInBestColumn,'we have a node:')
            newNode=treeNode('option',optionsInBestColumn)
            aNode.add_child(newNode)
            subNode=aNode.children[-1]
            checkMatrix(subMatrix,ColumnId,subNode)
        print ('we have finished with column',bestColumnId)
        ColumnId.append(bestColumnId)
        ColumnId.sort()



def desideRepOrDem(row,startNode):
    while (startNode.children[0].nodeType !='endPoint'):
        questionNumber=startNode.children[0].value
        questionNumberAnswer=row[questionNumber]
        startNode=startNode.children[0] #moving to the question in tree
        startNode=startNode.children[questionNumberAnswer] #moving to the answer we get from the row for that question
    return startNode.children[0].value



start = treeNode('start',0)
data= getXYMatrix()
ColumnId=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'y']#to know in which original column we are at

checkMatrix(data,ColumnId,start)


check=getMatrix('Xtest')
##print desideRepOrDem(check[68],start)
for row in range (len(check)):
    print desideRepOrDem(check[row],start)




exit(0)
