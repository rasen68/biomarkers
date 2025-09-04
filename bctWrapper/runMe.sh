#!/bin/bash
# sample: ./runMe.sh 
# Note that, depending on the dataset that you will run this script on, you might need to change the variable definitions below

experimentPath=.
dataset=rois_aal_NYU_Altered #TBI #make sure to put the name of the dataset correct here. Script will look for folder named with this name under "connectomes" and "src" folders
measuresList=$experimentPath/src/featureList.txt
connectomesFolder=$experimentPath/connectomes/$dataset
samplePath=$experimentPath/src/$dataset/subjects.txt
systemMaps=$experimentPath/src/$dataset/systemsMap.txt
hemisphereMaps=$experimentPath/src/$dataset/hemisphereMap.txt
numNodes=116 #86 #make sure that his number is the same as the number of nodes in the connectomes

python=python #you might need to change this if the python binary has name python or py on your machine rather than python3

bct_features_py=./scripts/bct_features.py

### If there does not exist a src/featuresList.txt file, generate a generic file which contains a full list of features that can be produced by the script. You need to rerun the script after that point to start producing these measures for the connectomes.
if [[ ! -e $measuresList ]]; then
	$python $bct_features_py --printFullMeasureList > $measuresList
	echo "Features list file is not found!! A full features lits is generated in $measuresList . Check the file and rerun this script to generate features for connectomes. Exiting..."
	exit 1
fi

mkdir -p $experimentPath/features

$python $bct_features_py --measuresList $measuresList --connectomesFolder $connectomesFolder/ --subjectsList $samplePath --systemMaps $systemMaps --hemisphereMaps $hemisphereMaps --outputFolder $experimentPath/features/ --numNodes $numNodes --verbose --normalizeConnectomes maxEdge1_subject


