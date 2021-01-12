from __future__ import division
import csv
import sys, os
import glob
import os.path
import struct
import numpy as np
import math
import copy
from numpy import linalg as li
import random
import pickle
from math import log, ceil, floor
import warnings
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split


#------------------------- gen_input -------------------------#

def gen_input():
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')
    def enablePrint():
        sys.stdout = sys.__stdout__
    ProteinAccessions = set()
    SequenceMD5digests = set()
    SequenceLengths = set() #Correlated
    Analyses = set()
    SignatureAccessions = set()
    SignatureDescriptions = set()
    StartLocations = set() #Correlated
    StopLocations = set() #Correlated
    Scores = set() #Correlated
    Status = set()
    Dates = set()
    InterProAnnoations_Accessions = set()
    InterProAnnotations_Descriptions = set()
    count = 0

    dataBuffer = []

    for filename in glob.glob('*.tsv'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            rd = csv.reader(f, delimiter="\t", quotechar='"')
            blockPrint()
            for row in rd:
                
                featureVector = []
                count+=1
                
                ProteinAccessions.add(row[0]) #cast types here
                #append to featureVector with appropriate type cast
                featureVector.append(row[0])

                SequenceMD5digests.add(row[1])
                #append to featureVector with appropriate type cast
                featureVector.append(row[1])

                SequenceLengths.add(float(row[2])) #float
                #append to featureVector with appropriate type cast
                try:
                    featureVector.append(float(row[2]))
                except:
                    print(row[2])

                Analyses.add(row[3])
                #append to featureVector with appropriate type cast
                featureVector.append(row[3])

                SignatureAccessions.add(row[4])
                #append to featureVector with appropriate type cast
                featureVector.append(row[4])

                SignatureDescriptions.add(row[5])
                #append to featureVector with appropriate type cast
                featureVector.append(row[5])

                StartLocations.add(float(row[6])) #float
                #append to featureVector with appropriate type cast
                try:
                    featureVector.append(float(row[6]))
                except:
                    print(row[6])

                StopLocations.add(float(row[7])) #float
                #append to featureVector with appropriate type cast
                try:
                    featureVector.append(float(row[7]))
                except:
                    print(row[7])

                #TODO: Figure this out
                
                #enablePrint()
                #print(row[8])

                raw_score = row[8]
                score = 0;
                if (raw_score == "-"):
                    score = 0;
                else:
                    score = float(raw_score)
                Scores.add(score)
                featureVector.append(score)

                #Scores.add(float(row[8])) # float
                #featureVector.append(float(row[8]))
                
                #blockPrint()

                Status.add(row[9])
                featureVector.append(row[9])

                Dates.add(row[10])
                featureVector.append(row[10])

                try:
                    InterProAnnoations_Accessions.add(row[11])
                    featureVector.append(row[11])
                except:
                    pass
                try:
                    InterProAnnotations_Descriptions.add(row[12])
                    featureVector.append(row[12])
                except:
                    pass

                dataBuffer.append(featureVector)
    
    #TODO: May neeed to iterate through dataBuffer to ensure correct typing

    features = []
    features.append(ProteinAccessions)
    features.append(SequenceMD5digests)
    features.append(SequenceLengths)
    features.append(Analyses)
    features.append(SignatureAccessions)
    features.append(SignatureDescriptions)
    features.append(StartLocations)
    features.append(StopLocations)
    features.append(Scores)
    features.append(Status)
    features.append(Dates)
    features.append(InterProAnnoations_Accessions)
    features.append(InterProAnnotations_Descriptions)


    enablePrint()
    return (features, count, dataBuffer)    

features, count, dataBuffer = gen_input()
# substitute count with len(dataBuffer)


#2. Create isCorrelated : boolean array
isCorrelated = [False for i in range(13)]
isCorrelated[2] = True
isCorrelated[6] = True
isCorrelated[7] = True
isCorrelated[8] = True
#isCorrelated[6:8] = True

# main
#D = 10000

#initialize HVs as an empty list
#HVs = []
#initialize HVList
#HVList = []
#for i in range(len(features)):
#    if(isCorrelated[i]):        #if a feature is correlated
#        LevelList = getlevelList(data_buffer, 100, i)
#        LevelHVs = genLevelHVs(100, D)
#        HVs.append(LevelHVs)
#        HVList.append(LevelList)
#    else :
        #    6. GenIDLists: dictionary that maps text to an index of a HV in HVs
#        IDList = genIDList(features[i])
        #    7. genIDHVs again #for encoding text values   
#        IDHVs = genIDHVs(size(features[i]), D) #totalPos = size of set
#        HVs.append(IDHVs)
#        HVList.append(IDList)

#IDHVs = genIDHVs(len(features), D) #13

#encodedBuffer = []
#for i in range(len(dataBuffer)):
#    encodedBuffer.append(IDMultHV(dataBuffer, D, Hvs, HVLists, isCorrelated))

#------------------------- genIDHVs -------------------------#

#Generates the ID hypervector dictionary
#Inputs:
#   totalPos: number of feature positions
#   D: dimensionality
#Outputs:
#   IDHVs: ID hypervector dictionary 
def genIDHVs(totalPos, D):
    print ('generating ID HVs')
    IDHVs = dict()
    indexVector = range(D)
    change = int(D / 2)
    for level in range(totalPos):
        name = level
        base = np.full(D, -1) #baseVal = -1
        toOne = np.random.permutation(indexVector)[:change]  
        for index in toOne:
            base[index] = 1
        IDHVs[name] = copy.deepcopy(base)     
    return IDHVs


#------------------------- Encoding -------------------------#

#Encodes a single datapoint into a hypervector
#Inputs:
#   inputBuffer: data to encode --> dataBuffer
#   D: dimensionality
#   IDHVs: ID hypervector bank (1 for each feature)
#   HVs: 3D HV list consisting of level hypervector banks and IDHV banks size:(nFeatures, varrys for each bank, D)
#   HVLists: list used to look up corresponding level HV or IDHV for the given feature value (also given an extra dimension as you'll have more of these)
#   isCorrelated: list of feature positions that are correlated (use level HVs)
#Outputs:
#   sumHV: encoded data
def IDMultHV (inputBuffer, D, IDHVs, HVs, HVLists, isCorrelated):
    #Initialize the output to zero
    sumHV = np.zeros(D, dtype = np.int)
    #Loop through input buffer features
    for i in range(len(inputBuffer)):
        #Get the IDHV for the current feature
        IDHV = IDHVs[i]
        #get level HV if feature is correlated
        if (i in isCorrelated):
            #find corresponding level HV for the feature value
            key = numToKey(inputBuffer[i], HVLists[i])
            HV = HVs[i][key]
        #if not correlated, must be independent (uses IDHV bank)
        else:
            #find corresponding IDHV for the feature value (can use dictionaries you built to do this) 
            #essentially mapping each unique value in that feature to an index
            key = IDnumToKey(inputBuffer[i], HVLists[i])
            HV = HVs[i][key]
        #multiply IDHV for the current feature with looked up level or IDHV and accumilate to the overall encoded HV
        sumHV = sumHV + (IDHV * HV)
    return sumHV





#------------------------- HDFunctions -------------------------#


#Splits up the feature value range into level hypervector ranges
#Inputs:
#   buffers: data matrix, n x 14 <--- samples x features
#   totalLevel: number of level hypervector ranges
#   n: particular (correlated) feature
#Outputs:
#   levelList: list of the level hypervector ranges
def getlevelList(buffers, totalLevel, n):
    try :
        minimum = min(buffers[:][n])
    except :
        print(n)
    maximum = max(buffers[:][n])
    length = maximum - minimum
    gap = length / totalLevel
    for lv in range(totalLevel):
        levelList.append(minimum + lv*gap)
    levelList.append(maximum)
    return levelList


#Generates the level hypervector dictionary
#Inputs:
#   totalLevel: number of level hypervectors
#   D: dimensionality
#Outputs:
#   levelHVs: level hypervector dictionary
def genLevelHVs(totalLevel, D):
    print ('generating level HVs')
    levelHVs = dict()
    indexVector = range(D)
    nextLevel = int((D/2/totalLevel))
    change = int(D / 2)
    for level in range(totalLevel):
        name = level
        if(level == 0):
            base = np.full(D, baseVal)
            toOne = np.random.permutation(indexVector)[:change]
        else:
            toOne = np.random.permutation(indexVector)[:nextLevel]
        for index in toOne:
            base[index] = base[index] * -1
        levelHVs[name] = copy.deepcopy(base)
    return levelHVs   

#A dictionary that maps an index to a unique value for features that aren't correlated.
#Inputs:
#   feature: a set
#   D: dimensionality
#Outputs:
#   IDList: a dictionary
def genIDList(feature):
    print ('generating ID list')
    IDList = dict()
    #count = 0
    #for val in feature
    #TODO: Check that the following is correct!
    for id,val in enumerate(feature):
        #IDList(val) = count
        #IDList[count].append(val)
        IDList.setdefault(id, val)
        #count+=1
    return IDList


#------------------ main -----------------------#
D = 10000

#initialize HVs as an empty list
HVs = []
#initialize HVList
HVList = []
for i in range(len(features)):
    if(isCorrelated[i]):        #if a feature is correlated
        LevelList = getlevelList(dataBuffer, 100, i)
        LevelHVs = genLevelHVs(100, D)
        HVs.append(LevelHVs)
        HVList.append(LevelList)
    else :
        #    6. GenIDLists: dictionary that maps text to an index of a HV in HVs
        IDList = genIDList(features[i])
        #    7. genIDHVs again #for encoding text values
        IDHVs = genIDHVs(len(features[i]), D) #totalPos = size of set
        HVs.append(IDHVs)
        HVList.append(IDList)

IDHVs = genIDHVs(len(features), D) #13

encodedTrain = []
for i in range(len(X_train)):
    encodedTrain.append(IDMultHV(X_train[i], D, Hvs, HVLists, isCorrelated))
    

encodedTest = []
for i in range(len(X_test)):
    encodedTest.append(IDMultHV(X_test[i], D, Hvs, HVLists, isCorrelated))
pickle.dump(encodedTrain, "encodedTrain.pkl")
pickle.dump(encodedTest, "encodedTest.pkl")
#----------------script.py----------------------#


# Processing .txt file
id2function = {}
txtFile = open('uniprot_sprot_exp.txt', 'r')
for line in txtFile:
    temp = re.split(r'\t+', line)
    try:
        ID = temp[0].strip()
        Function = temp[1]
        Function = re.split(r'\:', Function)
        Function = Function[1].strip()
        id2function[ID] = Function
    except:
        print(temp)
txtFile.close()

f = open("uniprot_sprot_exp.fasta.tsv", "r")
test = open("test.txt", "x")
functionCount = {}
for line in f:
    temp = line
    #Examine ID:
    ID = temp.split('\t')[0]
    #Lookup corresponding function
    try:
        Function = id2function[ID]
        #Prepend function identifier to row
        test.write(Function+'\t'+temp)
        #Increment count
        if Function in functionCount:
            functionCount[Function] +=1
                            
        else:
            functionCount[Function] = 1
            
    except:
        print(ID + " not found!")
    
f.close()
test.close()

#Identify most frequent element:
v=list(functionCount.values())
k=list(functionCount.keys())
maxElement = k[v.index(max(v))]
print(maxElement)

#Change labels:
f = open("test.txt", "r")
labelled = open("labels.txt", "x")
labelled.write('na\tna\tna\tna\tna\tna\tna\tna\tna\tna\tna\tna\tna\tna\tna\n')
for line in f:
    temp = line

if temp.split('\t')[0] == maxElement:
    labelled.write('1'+'\t'+line)

else:
    labelled.write('0'+'\t'+line)


f.close()
labelled.close()


import pandas as pd
import numpy as np

# make a dummy .tsv file, save it to disk
fpath = "/root/ProteinFunctionPrediction/data/CAFA3_training_data/labels.txt"
df=pd.read_csv(fpath, sep='\t')
df.to_csv("check.tsv", sep="\t")
f = open("check.tsv", "r")
data = df[["na.2","na.3","na.4","na.5","na.6","na.7","na.8","na.9","na.10","na.11","na.12","na.13","na.14"]]
labels = df[["na"]]
#print(data.head())
#print(labels.head())
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)
print(len(X_train))
print(len(X_test))
print(len(data))
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())
