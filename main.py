from utils import getCifar10Data, getMnistData, getFMnistData, saveAsPickle,getTinyImagenetData
from algorithms.ghcidr import GHCIDR
from algorithms.baselineRHC import RHC
from algorithms.mghcidr import Merged_GHCIDR
import argparse
import os,sys
parser = argparse.ArgumentParser()
parser.add_argument("-variantName", default='MergeGHCIDR', type=str)
parser.add_argument("-datasetName", default='MNIST', type=str)
parser.add_argument("-storeClusters", default=True, type=bool)
parser.add_argument("-savePicklePath", default='../reducedData/')
parser.add_argument("-alpha", default=0.3, type=float) 
parser.add_argument("-beta", default=0.3, type=float) 
parser.add_argument("-KFarthest", default=1, type=int)
args = parser.parse_args()

Datasets = {
            "MNIST":getMnistData,
            "FMNIST":getFMnistData, 
            "CIFAR10":getCifar10Data,
            "TinyImagenet":getTinyImagenetData
            }
Variants = {
            "GHCIDR":GHCIDR,
            "RHC":RHC,
            "MGHCIDR": Merged_GHCIDR
            }

dataCall, variantCall = Datasets[args.datasetName],Variants[args.variantName]
reducedData = variantCall(dataCall(),args)
if(args.variantName == "MGHCIDR"):
    path =  "./datasetPickle/" +args.datasetName + '_' + args.variantName + '_'+ str(args.beta)+".pickle"
else:
    path =  "./datasetPickle/" +args.datasetName + '_' + args.variantName +".pickle"
saveAsPickle(reducedData,path)

