import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from utils import *
from .baselineRHC import homogeneousClustering
from .ghcidr import GHCIDR


def mergeHomogeneousClusters(unique_Classes,Clusters,args):
    ImagesClass = defaultdict(list)
    for i in range(unique_Classes):
        ImagesClass[i] = []
    for i in Clusters:
        ImagesClass[i[1]].append(i[0])
    NewClusters = []
    Beta = args.beta
    ans = 0
    for j in range(unique_Classes):
        Centroids = []
        for i in ImagesClass[j]:
            Centroids.append(np.mean(i,axis=0))
        l = len(ImagesClass[j])
        Centroids = np.array(Centroids)
        x = AgglomerativeClustering(n_clusters=int(l*Beta),linkage="complete").fit(Centroids).labels_.tolist()
        Final = {}
        for i in range(int(l*Beta)):
            Final[i] = []
        for i,cluster in enumerate(ImagesClass[j]):
            Final[x[i]].extend(cluster)
        for i in Final.keys():
            Final[i] = np.array(Final[i])
        addit = list(Final.values())
        for i in range(len(addit)):
            NewClusters.append([addit[i],j])
    if "MergedClusters" not in os.listdir():
        os.mkdir("MergedClusters")
    path = "./MergedClusters/"+args.datasetName + "_" + str(args.beta) +'.pickle'
    saveAsPickle(NewClusters,path)
    print("MergedClusters are saved")

def Merged_GHCIDR(datasets,args):
    X_train,Y_train = datasets
    if(checkClusters(args)==False):
        print("Making Homogeneous Clusters.....")
        homogeneousClustering(X_train,Y_train,args)
    print("/homogeneous clusters are done")
    Clusters = loadFromPickle("./Clusters/"+args.datasetName+".pickle")
    if(checkMergedClusters(args)==False):
        unique_Classes = len(np.unique(Y_train))
        print("Merging Clusters")
        mergeHomogeneousClusters(unique_Classes,Clusters,args)
    return GHCIDR(datasets,args,mode=1)









