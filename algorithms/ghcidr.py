import pickle
from .baselineRHC import homogeneousClustering
from utils import *
# def GHCIDR(X_train,X_test,Y_train,Y_test,max_size,givenClusters):
def GHCIDR(datasets,args,mode=0):
    """
        GHCIDR algorithm
        First gets all the homogenous cluters
        Then applies GHCIDR algorithm on these clusters
    """
    X_train,Y_train = datasets
    if(mode==0):
        if checkClusters(args)==False:
            homogeneousClustering(X_train,Y_train,args)
        Clusters = loadFromPickle("./Clusters/"+args.datasetName+".pickle")
    else:
        Clusters = loadFromPickle("./MergedClusters/"+args.datasetName + "_" + str(args.beta) +".pickle")
    CondensedSet = []
    print("Running GHCIDR")
    for i in Clusters:
        cImages = i[0]
        meanVector = np.mean(cImages,axis=0)
        distances = []
        dis = []
        for j in cImages:
            r = getL2NormDistnce(meanVector,j)
            distances.append([r,j])
            dis.append(r)
        distances.sort(key=itemgetter(0))
        maxDist = distances[-1][0]
        beta = int(max(1,maxDist//((1-args.alpha)*len(cImages))))
        for j in range(0,4*int(max(dis))//5,beta):
            arr = []
            for k in range(1,len(distances)):
                if distances[k][0]>j and distances[k][0]<j+1:
                    arr.append(k)
            if arr:
                CondensedSet.append([distances[arr[-1]][1],i[1]])
        for j in range(4*int(max(dis))//5,int(max(dis))+1,1):
            arr = []
            for k in range(1,len(distances)):
                if distances[k][0]>j and distances[k][0]<j+1:
                    arr.append(k)
            if arr:
                CondensedSet.append([distances[arr[-1]][1],i[1]])
        CondensedSet.append([distances[0][1],i[1]])
    return CondensedSet