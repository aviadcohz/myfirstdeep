import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import gmtime, strftime
import time


# a row with 28^2 pixs
def printARow(rowToPrint):
    matplotlib.interactive(True)
    matplotlib.use("TKAgg")
    plt.imshow(rowToPrint.reshape(28,28),cmap="gray");
    plt.draw();
##    time.sleep(0.5)
##    plt.close()


def loadTraining():
    #train - 60,000 examples, 784 pixels per example
    x = np.loadtxt("Xtrain.gz",dtype=int);#,skiprows=59900
    y = np.loadtxt("Ytrain.gz",dtype=int);
    return x,y


#---------------training-----------------------

def getW(x,y,l,T):
    w=np.zeros((k,len(x[0])))#10X784
    wsum=np.zeros((k,len(x[0])))#10X784
    for t in range(T):
        i=np.random.randint(len(x))#take one of the examples
        res=[]#here we will put the dot product of all k options
        for number in range(k):
            res.append(np.dot(w[number],x[i]))
        realAns=res[y[i]]#this will be w*psi(xi,realAns y)
        res+=1-realAns
        res[y[i]]-=1#this is for the delta(y,realAns y), if y is realAns y then we should remove the 1 that we added before
        ymax=res.argmax()
        v=l*w
        v[ymax]+=x[i]
        v[y[i]]-=x[i]
        w-=v/(l*(t+1))
        wsum+=w
    wsum/=T
    return wsum



#------------------testing-----------------------
def loadTest():
    #train - 60,000 examples, 784 pixels per example
    x = np.loadtxt("Xtest.gz",dtype=int);#10,000 examples
    y = np.loadtxt("Ytest.gz",dtype=int);
    return x,y

def checkTest(x,y,w):
    error=0
    for xval,yval in zip(x,y):
        res=[]#here we will put the dot product of all k options
        for number in range(len(w)):
            res.append(np.dot(w[number],xval))
        if res.index(max(res)) != yval:
            error+=1
##            printARow(xval)
##            print "this is ",yval," and w got ",res.index(max(res))
##            time.sleep(2)
##            plt.close()
    return 100.0*error/len(x)



#---------------main---------------------------

x,y=loadTraining()
T=10*len(x)
k=10
print "done reading the training data"
for l in [1,0.1,0.01]:
    print "lambda is ",l
    w=getW(x,y,l,T)
    print "done finding w"
    x,y=loadTest()
    print "done reading the test data"
    e=checkTest(x,y,w)
    print"error is: ",e,"%"




