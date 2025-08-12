#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:56:39 2020

@author: yusuf
"""
import argparse
import numpy as np
from numpy.lib.recfunctions import append_fields,rename_fields,drop_fields

## --subjectsList /home/yusuf/repo/projects/tbiStructureAndFunction/results/brainLes++/data/src/subjects_qa.txt --measuresFolder /home/yusuf/repo/projects/tbiStructureAndFunction/results/brainLes++/08_bctFeatures/features/raw_gmvmivol --measuresList /home/yusuf/repo/projects/tbiStructureAndFunction/results/brainLes++/08_bctFeatures/featureList.txt --similarityScores /home/yusuf/repo/projects/tbiStructureAndFunction/results/brainLes++/01_matching/results/processed/raw_gmvmivol/direct/connectomeLevel/accuracy_healthy_direct.res --outputFolder /home/yusuf/repo/projects/tbiStructureAndFunction/results/brainLes++/08_bctFeatures/raw_gmvmivol_

######################main code####################
##### get command line parameters
parser = argparse.ArgumentParser(description='Gather values for a random variable in male/female populations')
parser.add_argument('-ml','--measuresList', help='path to the file that contains list of graph theory measures to be calculated', required=True,type=str,default="")
parser.add_argument('-m','--measure', help='name of the single graph theory measures to be calculated', required=False,type=str,default="")
parser.add_argument('-mf','--measuresFolder', help='folder path that contains the node features per subject for multiple feature types', required=True)
parser.add_argument('-sl','--subjectsList', help='file path containing subjectIDs', required=True)
parser.add_argument('-d','--demographics', help='file path containing demographics info and cognitive scores', required=True)
parser.add_argument('-out','--outputPath', help='file path to save the distribution of values for the two populations', required=True)

args = vars(parser.parse_args())
measuresListPath=args['measuresList']
singleMeasure=args['measure']
measuresFolder=args['measuresFolder']
subjectListPath=args['subjectsList']
outputPath=args['outputPath']
demographicsPath=args['demographics']

#########load subject list
subjectList=np.loadtxt(subjectListPath,dtype=str).tolist()

############### update the subjectsInfo table by dropping the unnecessary columns and renaming the rest of the columns to fit my standards
#subjectsInfo=rename_fields(subjectsInfo,{'Subject':'subjectId','Group':'timePoint','Gender':'gender','PSIT': 'PS', 'ReyT':'VL', 'PTA_Estimated':'PTA','Age':'age','DaysSinceInjury':'DSI'})
#subjectsInfo=drop_fields(subjectsInfo, ['Status', 'GCS_Total', 'PTA_Duration', 'Days_to_follow_commands', 'Education_Coded', 'Trails_B__TScore', 'DSBT', 'LNST', 'COWAT', 'CWITT'])

######################load graph theory measures
###below are all possible global measures that the code generates
globalMeasures=["degree_avg","degree_interHemisphere_avg","degree_intraHemisphere_avg","degree_withinModule_avg","degree_betweenModule_avg",
                "strength_global","strength_global_offDiagonal","strength_interHemisphere_global","strength_intraHemisphere_global","strength_selfConnections_global",
                "node_betweenness_centrality_avg","eigenvector_centrality_avg","participation_coefficient_avg",
                "clustering_coefficient_avg","clustering_coefficient_zhang_avg","eccentricity_avg",
                "small_worldness","characteristic_path_length","global_efficiency","radius","diameter","modularity_global","assortativity","density",
                "degree_neg_avg","degree_interHemisphere_neg_avg","degree_intraHemisphere_neg_avg","degree_withinModule_neg_avg","degree_betweenModule_neg_avg",
                "strength_global_neg","strength_global_offDiagonal_neg","strength_interHemisphere_global_neg","strength_intraHemisphere_global_neg",
                "participation_coefficient_pos_avg","participation_coefficient_neg_avg","clustering_coefficient_neg_avg","clustering_coefficient_zhang_neg_avg","modularity_global_neg"]
                
#########load list of graph theory measures to be calculated
with open(measuresListPath) as fl:
    loaded = fl.read().splitlines()
    measuresList = [x for x in loaded if x[0]!='#']

### sort out measures list into connectome/system/node level
measureNames=[]
for measure in measuresList:
    if(measure not in globalMeasures):
        print("Measure <%s> that was asked among measures to calculate cannot be processed by the script as it's not recognized by the script. Execution will continue with the rest of the measures (if any)." % measure)
        if(len(measuresList)==1): # if this was the only measure (which ended up being incorrect), then exit execution
            exit()
    elif(measure in globalMeasures):
        measureNames.append(measure)

# load connectome level (i.e., global) BCT measures
measures=[]
for measureName in measureNames:
    measures.append(np.loadtxt(measuresFolder+"/"+measureName+".txt"))
measures=np.array(measures) ### measures matrix is of dimension: (numMeasures x numSubjects)
numMeasures=len(measureNames)

####### save table into file
reportFile=open(outputPath,'w')
reportFile.write("ID")
for i in range(numMeasures):
    reportFile.write(",%s" %measureNames[i])
reportFile.write("\n")

for i in range(len(subjectList)):
    reportFile.write("%s" % (subjectList[i]))
    for j in range(numMeasures):
        reportFile.write(",%f" %measures[j][i])
    reportFile.write("\n")
reportFile.close()  
