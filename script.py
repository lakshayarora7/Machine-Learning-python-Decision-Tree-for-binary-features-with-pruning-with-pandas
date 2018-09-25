import sys
import pandas as pd
import math
import random
import copy
  
if(len(sys.argv) != 7):     # also includes sys.argv[0] i.e. file of name
    sys.exit("Please give the required amount of arguments - <L> <K> <training-set> <validation-set> <test-set> <to-print>")
#if False: pass
else:
    integerL = int(sys.argv[1])
    integerK = int(sys.argv[2])
    pathToTrainingFile = sys.argv[3]
    pathToValidationSet = sys.argv[4]
    pathToTestFile = sys.argv[5]
    toPrint = sys.argv[6]
#    integerL = 10
#    integerK = 10
#    pathToTrainingFile = r'D:\DOWNLOADS\CS6375.002.18F Machine Learning by Anjum Chida\4. Decision Trees\Assignment 1\data_sets2\training_set.csv'
#    pathToTestFile = r'D:\DOWNLOADS\CS6375.002.18F Machine Learning by Anjum Chida\4. Decision Trees\Assignment 1\data_sets2\test_set.csv'
#    pathToValidationSet = r'D:\DOWNLOADS\CS6375.002.18F Machine Learning by Anjum Chida\4. Decision Trees\Assignment 1\data_sets2\validation_set.csv'
#    toPrint = "no"

# Reading Files
trainingDataSet = pd.read_csv(pathToTrainingFile)          # [600 rows x 21 columns]
testingDataSet = pd.read_csv(pathToTestFile)               # [2000 rows x 21 columns]
validationDataSet = pd.read_csv(pathToValidationSet)       # [2000 rows x 21 columns]

# Removing records containing any null attribute value (NaN)
trainingDataSet = trainingDataSet.dropna()                  # [600 rows x 21 columns]
testingDataSet = testingDataSet.dropna()                    # [2000 rows x 21 columns]
validationDataSet = validationDataSet.dropna()              # [2000 rows x 21 columns]

def entropy(subset):
    totalNumberOfRecordsInSubset = subset.shape[0]
    numberOfOnesInClass = subset.sum().sum()                # 1st sum() -> Class 300, dtype: int64; 2nd sum() -> 300
    numberOfZeroesInClass = totalNumberOfRecordsInSubset - numberOfOnesInClass
    if totalNumberOfRecordsInSubset == numberOfOnesInClass or totalNumberOfRecordsInSubset == numberOfZeroesInClass:
        return 0
    entropy = -(numberOfOnesInClass/totalNumberOfRecordsInSubset)*math.log(numberOfOnesInClass/totalNumberOfRecordsInSubset, 2) - (numberOfZeroesInClass/totalNumberOfRecordsInSubset)*math.log(numberOfZeroesInClass/totalNumberOfRecordsInSubset,2)
    return entropy

# print(entropyCalculator(trainingDataSet[['Class']]))

def informationGainUsingEntropy(comparisonLabels):         # comparisonLabels = trainingDataSet[['XB', 'Class']] 
    '''
    comparisonLabels contain portion of dataFrame with only 2 lables, one is the label being analysed, the other is 'Class'.
    
    ###### ENTROPY #########
    E(S) read as entropy of S relative to its boolean classification of positive and negative examples
                = - proportion of positive examples in S * log 2 (proportion of positive examples in S)
                  - proportion of negative examples in S * log 2 (proportion of negative examples in S)
    
    Values(Wind) = Weak, Strong          # Wind is an attribute/label with values Weak and Strong
    S = [9+, 5-]                         # Total number of examples in training set
    S(weak) = [6+, 2-]                   # Subset of S with Wind = weak
    S(strong) = [3+, 3-]                 # Subset of S with Wind = strong
    
    E(weak) = -(6/8)*log2(6/8)] -(2/8)*log2(2/8)
    E(strong) = -(3/6)*log2(3/6)] -(3/6)*log2(3/6)
    
    ####### INFORMATION GAIN #######
    Gain(S, Wind) read as 'The information gain of attribute Wind relative to a collection of samples S
                  = E(S) - (total number of examples in S with value weak / total number of examples in training set) * E(weak)
                                - (total number of examples in S with value strong / total number of examples in training set) * E(strong)
                  = E(S) - (8/14) E(weak) - (6/14) E(strong)
    '''
    overallEntropy = entropy(comparisonLabels[['Class']]) # 1.0
    totalNumberOfRecordsInSubset = comparisonLabels.shape[0]   # [600 rows x 2 columns] -> 600
    
    labelOne = comparisonLabels.columns[0]      # 'XB'; labelTwo is 'Class'    
    # S(0): portion of comparisonLabels dataframe with value of lableOne set to 1
    subsetOneOfComparisonLabels = comparisonLabels[comparisonLabels[labelOne] == 1]     # [265 rows x 2 columns]
    numberOfExWithAttrValueOne = subsetOneOfComparisonLabels.shape[0]                   # 265 for 'XB'
    # S(1): portion of comparisonLabels dataframe with value of lableOne set to 0
    subsetZeroOfComparisonLabels = comparisonLabels[comparisonLabels[labelOne] == 0]    # [335 rows x 2 columns]
    numberOfExWithAttrValueZero = subsetZeroOfComparisonLabels.shape[0]                 # 335 for 'XB'
    
    # Entropy of subsetOne relative to + and - classification of Class
    entropyOfOne = entropy(subsetOneOfComparisonLabels['Class']) # sends values of Class for all 265 records for which value of XB is 1
    # Entropy of subsetZero relative to + and - classification of Class
    entropyOfZero = entropy(subsetZeroOfComparisonLabels['Class']) # sends values of Class for all 335 records for which value of XB is 0
    
    infoGain = overallEntropy - (numberOfExWithAttrValueOne/totalNumberOfRecordsInSubset)*entropyOfOne - (numberOfExWithAttrValueZero/totalNumberOfRecordsInSubset)*entropyOfZero
    return infoGain

def findBestAttributeUsingEntropy(data):
    minimumInformationGain = -1.0
    for label in data.columns:
        if label == 'Class':
            continue
        currentInfoGain = informationGainUsingEntropy(data[[label, 'Class']])
        if minimumInformationGain < currentInfoGain:
            minimumInformationGain = currentInfoGain
            bestAttributeUsingIG = label
    return bestAttributeUsingIG

def variance(subset):
    totalNumberOfRecordsInSubsetK = subset.shape[0]         # K
    numberOfOnesInClassK1 = subset.sum().sum()              # K1
    numberOfZeroesInClassK0 = totalNumberOfRecordsInSubsetK - numberOfOnesInClassK1    # K0
    if totalNumberOfRecordsInSubsetK == numberOfOnesInClassK1 or totalNumberOfRecordsInSubsetK == numberOfZeroesInClassK0:
        return 0
    variance = (numberOfOnesInClassK1 / totalNumberOfRecordsInSubsetK) * (numberOfZeroesInClassK0 / totalNumberOfRecordsInSubsetK)
    return variance

def informationGainWithVariance(comparisonLabels):
    '''
    comparisonLabels contain portion of dataFrame with only 2 lables, one is the label being analysed, the other is 'Class'.
    
    ###### VARIANCE #########
    V(K) read as variance of K relative to its boolean classification of positive and negative examples
                = proportion of positive examples in K * proportion of negative examples in K
    
    K: total number of examples in comparisonLabels datafram
    K(0): portion of comparisonLabels dataframe with value of Class set to 0
    K(1): portion of comparisonLabels dataframe with value of Class set to 1
    V(K) = (K0/K) * (K1/K)
    
    Values(Wind) = Weak, Strong          # Wind is an attribute/label with values Weak and Strong
    K = [9+, 5-]                         # Total number of examples in training set
    K(weak) = [6+, 2-]                   # Subset of K with Wind = weak
    K(strong) = [3+, 3-]                 # Subset of K with Wind = strong
    
    V(weak) = (6/8)*(2/8)
    V(strong) = (3/6)*(3/6)
    
    ####### INFORMATION GAIN #######
    Gain(K, Wind) read as 'The information gain of attribute Wind relative to a collection of samples K
                  = V(K) - (total number of examples in K with value weak / total number of examples in training set) * V(weak)
                                - (total number of examples in K with value strong / total number of examples in training set) * V(strong)
                  = V(K) - (8/14) V(weak) - (6/14) V(strong)
    '''
    overallVariance = variance(comparisonLabels[['Class']])     # 300/600 * 300/600 = 0.25
    totalNumberOfRecordsInSubsetK = comparisonLabels.shape[0]   # [600 rows x 2 columns] -> 600 # K
  
    labelOne = comparisonLabels.columns[0]      # 'XB'; labelTwo is 'Class'    
    # K(0): portion of comparisonLabels dataframe with value of Class set to 0
    subsetOneOfComparisonLabels = comparisonLabels[comparisonLabels[labelOne] == 1]     # [265 rows x 2 columns]
    numberOfExWithAttrValueOne = subsetOneOfComparisonLabels.shape[0]                   # 265 for 'XB'
    # K(1): portion of comparisonLabels dataframe with value of Class set to 1
    subsetZeroOfComparisonLabels = comparisonLabels[comparisonLabels[labelOne] == 0]    # [335 rows x 2 columns]
    numberOfExWithAttrValueZero = subsetZeroOfComparisonLabels.shape[0]                 # 335 for 'XB'
      
    # Variance of subsetOne relative to + and - classification of Class
    varianceOfOne = variance(subsetOneOfComparisonLabels['Class']) 
    # Variance of subsetZero relative to + and - classification of Class
    varianceOfZero = variance(subsetZeroOfComparisonLabels['Class']) 
    
    informationGain = overallVariance - (numberOfExWithAttrValueOne/totalNumberOfRecordsInSubsetK)*varianceOfOne - (numberOfExWithAttrValueZero/totalNumberOfRecordsInSubsetK)*varianceOfZero
    return informationGain                                                      

def findBestAttributeUsingVariance(data):
    minimumInformationGain = -1.0                       
    for label in data.columns:
        if label == 'Class':
            continue
        currentInformationGain = informationGainWithVariance(data[[label, 'Class']])
        if minimumInformationGain < currentInformationGain:                        
            minimumInformationGain = currentInformationGain
            bestAttributeUsingVariance = label
    return bestAttributeUsingVariance

class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.nodeType = None        # L/R/I leaf/Root/Intermidiate 
        self.value = None           # attributes split's value 0 or 1
        self.positiveCount = None
        self.negativeCount = None
        self.label = None
        self.nodeId = None
    
    def setNodeValues(self, attribute, nodeType, value = None, positiveCount = None, 
                     negativeCount = None):
        self.attribute = attribute
        self.nodeType = nodeType
        self.value = value
        self.positiveCount = positiveCount
        self.negativeCount = negativeCount

class Tree():
    def __init__(self):
        self.root = Node()
        self.root.setNodeValues('default', 'R')
        
    def createDecisionTree(self, data, tree, entropyOrVariance):
        global nodeCount
        totalNumberOfRecordsInDataSet = data.shape[0]       # df.shape = (600,,21); shape[0] -> total number of records
        numberOfOnesInClass = data['Class'].sum()
        numberOfZeroesInClass = totalNumberOfRecordsInDataSet - numberOfOnesInClass        
        if data.shape[1] == 1 or totalNumberOfRecordsInDataSet == numberOfOnesInClass or totalNumberOfRecordsInDataSet == numberOfZeroesInClass:
            tree.nodeType = 'L'
            if numberOfZeroesInClass >= numberOfOnesInClass:    # setting most common value 
                tree.label = 0
            else:
                tree.label = 1
            return        
        else:
            if entropyOrVariance == 'E':
                bestAttribute = findBestAttributeUsingEntropy(data)
            elif entropyOrVariance == 'V':
                bestAttribute = findBestAttributeUsingVariance(data)
            tree.left = Node()
            tree.right = Node()
            
            tree.left.nodeId = nodeCount
            nodeCount = nodeCount + 1
            tree.right.nodeId = nodeCount
            nodeCount = nodeCount + 1
            
            tree.left.setNodeValues(bestAttribute, 'I', 0, data[(data[bestAttribute]==0) & (data['Class']==1) ].shape[0], data[(data[bestAttribute]==0) & (data['Class']==0) ].shape[0])
            tree.right.setNodeValues(bestAttribute, 'I', 1, data[(data[bestAttribute]==1) & (data['Class']==1) ].shape[0], data[(data[bestAttribute]==1) & (data['Class']==0) ].shape[0])
            self.createDecisionTree( data[data[bestAttribute]==0].drop([bestAttribute], axis=1), tree.left, entropyOrVariance)
            self.createDecisionTree( data[data[bestAttribute]==1].drop([bestAttribute], axis=1), tree.right, entropyOrVariance)
            
    def printTreeHelper(self, node, step):
        if(node.left is None and node.right is not None):
            for i in range(0, step):    
                print("| ", end="")             # end = "" prevents the print() to insert a \n character
            step = step + 1
            print("{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else "")))
            self.printTreeHelper(node.right, step)
        elif(node.right is None and node.left is not None):
            for i in range(0, step):    
                print("| ", end="")
            step = step + 1
            print("{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else "")))
            self.printTreeHelper(node.left, step)
        elif(node.right is None and node.left is None):
            for i in range(0, step):    
                print("| ", end="")
            step = step + 1
            print("{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else "")))
        else:
            for i in range(0, step):    
                print("| ", end="")
            step = step + 1
            print("{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else "")))
            self.printTreeHelper(node.left, step)
            self.printTreeHelper(node.right, step)
    
    def printTree(self, node):
        self.printTreeHelper(node.left, 0)
        self.printTreeHelper(node.right, 0)
    
    def classifyUnseenExample(self, data, root):
        '''goal of this function is to return a label (i.e. value of class 0 or 1); 
        labels are at leaf nodes only.
        
        root = decisionTreeD.root
        print(root.right.attribute, root.right.value)               # XO 1
        print(root.left.attribute, root.left.value)                 # XO 0
        rightNode1 = root.right
        print(rightNode1.right.attribute, rightNode1.right.value)   # XI 1
        print(rightNode1.left.attribute, rightNode1.left.value)     # XI 0
        leftNode1 = root.left
        print(leftNode1.right.attribute, leftNode1.right.value)     # XM 1
        print(leftNode1.left.attribute, leftNode1.left.value)       # XM 0

        rightNode2 = rightNode1.right
        print(rightNode2.right.attribute, rightNode2.right.value)   # XT 1
        print(rightNode2.left.attribute, rightNode2.left.value)     # XT 0
        '''
        
        # if at a leaf node, return the label
        if root.label is not None:          
            return root.label
        # else, if value for data['XB'][0] (for index 0 i.e. 1st training example) is 1,
        # then traverse the right part of the tree; reason for this is, that we when we
        # are createing the decision tree, we are setting value 1 to right nodes (as 
        # demonstrated above interpreter examples), and 
        # 0 to left nodes. Basically, we want a string of 1s till we get to a leaf node, and 
        # then return it label.
        elif data[root.left.attribute][data.index.tolist()[0]] == 1:  # essentially data['XB'][0] == 1 or not
            return self.classifyUnseenExample(data, root.right)
        else:
            return self.classifyUnseenExample(data, root.left)

    def countNonLeafNodes(self,node):
        if(node.left is not None and node.right is not None):
            return self.countNonLeafNodes(node.left) + self.countNonLeafNodes(node.right) + 2
        return 0
    
#    def countNonLeafNodes(self, node):
#        if (node.left is not None and node.right is not None):
#            return self.countNonLeafNodes(node.left) + self.countNonLeafNodes(node.right) + 2
#        elif (node.left is not None and node.right is None):
#            return self.countNonLeafNodes(node.left) + 1
#        elif (node.left is None and node.right is not None):
#            return self.countNonLeafNodes(node.right) + 1
#        else:
#            return 0

#    def countAllLeafNodes(self,node):
#        if(node.left is None and node.right is None):
#            return 1
#        return self.countAllLeafNodes(node.left) + self.countAllLeafNodes(node.right)

def findNode(tree, integerP):
    nodeFoundOrNot = None
    foundTree = None
    
    # If node is not a leaf node
    if(tree.nodeType != "L"):
        # if node with nodeId = integerP found, then return the tree
        if(tree.nodeId == integerP):
            return tree
        # else, search the left and right parts of the tree recursively
        else:
            foundTree = findNode(tree.left,integerP)
            if (foundTree is None):
                foundTree = findNode(tree.right,integerP)
            return foundTree
    # else, return the temporary node with default value None
    else:
        return nodeFoundOrNot

def pruningFunction(integerM, newTree):
    for integerJ in range(1, integerM + 1):
        countOfNonLeafNodes = newTree.countNonLeafNodes(newTree.root)
        integerN = countOfNonLeafNodes
        integerP = random.randint(1, integerN)
        
        lookedUpNode = Node()
        lookedUpNode = findNode(newTree.root, integerP)

        if(lookedUpNode is not None):
            lookedUpNode.left = None
            lookedUpNode.right = None
            lookedUpNode.nodeType = "L"     # Replacing the subtree rooted at lookedUpNode by a Leaf Node and assigning the majority value # The printTreeHelper() will automatically take care of cutting the tree.
            if(lookedUpNode.negativeCount >= lookedUpNode.positiveCount):
                lookedUpNode.label = 0
            else:
                lookedUpNode.label = 1
        return newTree

def accuracyCalculator(data, tree):
    correctlyClassified = 0
    for ind in data.index:
        value = tree.classifyUnseenExample(data.iloc[ind:ind + 1, :].drop(['Class'], axis=1), tree.root)
        if value == data['Class'][ind]:
            correctlyClassified = correctlyClassified + 1
    return round(correctlyClassified / data.shape[0] * 100, 2)


################## Entropy #########################
print("************ CREATING DECISION TREE BASED ON ENTROPY HEURISTIC, PLEASE WAIT (TAKES AROUND 30 SECONDS) **********")
nodeCount = 0                                               # for node id
decisionTreeD = Tree()
decisionTreeD.createDecisionTree(trainingDataSet, decisionTreeD.root, 'E')

maximumAccuracy = accuracyCalculator(validationDataSet, decisionTreeD)
bestTreeDbest = copy.deepcopy(decisionTreeD)
countOfNonLeafNodes = bestTreeDbest.countNonLeafNodes(bestTreeDbest.root)

if toPrint == 'yes':
    print("******DECISION TREE BEFORE PRUNING (ENTROPY HEURISTIC) *******")
    decisionTreeD.printTree(decisionTreeD.root)

print("\n****** STATS FOR TESTING DATASET BEFORE PRUNING FOR ENTROPY HEURISTIC ******")
print("Number of instances: {}".format(str(testingDataSet.shape[0])))
print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, decisionTreeD))))

print("\nDISTINCT COMBINATIONS OF L AND K FOR ENTROPY HEURISTIC:")
for integerI in range(1, integerL + 1):                    
    pruneTreeDdash = Tree()
    pruneTreeDdash = copy.deepcopy(bestTreeDbest)
    integerM = random.randint(1,integerK)
    pruneTreeDdash = pruningFunction(integerM, pruneTreeDdash)
    accuracyOfNewlyPrunedTree = accuracyCalculator(validationDataSet, pruneTreeDdash)
    print("L: {}, K: {}, Accuracy: {}".format(integerI, integerM, accuracyOfNewlyPrunedTree))
    if accuracyOfNewlyPrunedTree > maximumAccuracy:             # print("Accuracy Improved")
        valueOfLforBestTree = integerI
        valueOfKforBestTree = integerM
        maximumAccuracy = accuracyOfNewlyPrunedTree
        bestTreeDbest = copy.deepcopy(pruneTreeDdash)
        
if toPrint == 'yes':
    print("\n******DECISION TREE AFTER PRUNING (ENTROPY HEURISTIC) *******")
    bestTreeDbest.printTree(bestTreeDbest.root)
    print("********* FINISHED PRINTING DECISION TREE BASED ON ENTROPY HEURISTIC ************")
    
        
print("\n****** STATS FOR TESTING DATASET AFTER PRUNING FOR ENTROPY HEURISTIC ******")
print("Number of instances: {}".format(str(testingDataSet.shape[0])))
print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, bestTreeDbest))))

#print("\n****** STATS FOR ALL DATASETS BEFORE PRUNING FOR ENTROPY HEURISTIC ******")
#print("\n1. TRAINING DATASET")
#print("Number of instances: {}".format(str(trainingDataSet.shape[0])))
#print("Number of attributes: {}".format(str(trainingDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(trainingDataSet, decisionTreeD))))

#print("\n2. VALIDATION DATASET")
#print("Number of instances: {}".format(str(validationDataSet.shape[0])))
#print("Number of attributes: {}".format(str(validationDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(validationDataSet, decisionTreeD))))
#
#print("\n3. TESTING DATASET")
#print("Number of instances: {}".format(str(testingDataSet.shape[0])))
#print("Number of attributes: {}".format(str(testingDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, decisionTreeD))))

#print("\n******STATS FOR ALL DATASETS AFTER PRUNING FOR ENTROPY HEURISTIC******")
#print("\n1. TRAINING DATASET")
#print("Number of training instances: {}".format(str(trainingDataSet.shape[0])))
#print("Number of training attributes: {}".format(str(trainingDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(trainingDataSet, bestTreeDbest))))
#
#print("\n2. VALIDATION DATASET")
#print("Number of Instances: {}".format(str(validationDataSet.shape[0])))
#print("Number of attributes: {}".format(str(validationDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(validationDataSet, bestTreeDbest))))
#
#print("\n3. TEST DATASET")
#print("Number of instances: {}".format(str(testingDataSet.shape[0])))
#print("Number of attributes: {}".format(str(testingDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, bestTreeDbest))))

#print("\n")
#
#accuracyOnValidationSetUnpruned = accuracyCalculator(validationDataSet, decisionTreeD)
#if(maximumAccuracy > accuracyOnValidationSetUnpruned):
#    print("Pruned successfully when L: {}, K: {}. Total pruning attempts: {}".format(valueOfLforBestTree, valueOfKforBestTree, integerL))
#    print("Pre-pruning accuracy: {}%; Post-pruning accuracy: {}%".format(accuracyOnValidationSetUnpruned, maximumAccuracy))
#else:
#    print("Decision Tree pruned. However, accuracy didn't improve after {} attempts, given by value of integer L.".format(integerL))    
    
    
################## Variance #########################
print("\n\n*************CREATING DECISION TREE BASED ON VARIANCE HEURISTIC, PLEASE WAIT (TAKES AROUND 30 SECONDS) ************")
nodeCount = 0                                               # for node id
decisionTreeD = Tree()
decisionTreeD.createDecisionTree(trainingDataSet, decisionTreeD.root, 'V')

maximumAccuracy = accuracyCalculator(validationDataSet, decisionTreeD)
bestTreeDbest = copy.deepcopy(decisionTreeD)
countOfNonLeafNodes = bestTreeDbest.countNonLeafNodes(bestTreeDbest.root)

if toPrint == 'yes':
    print("******PRINTING DECISION TREE BEFORE PRUNING (VARIANCE HEURISTIC) *******")
    decisionTreeD.printTree(decisionTreeD.root)

print("\n****** STATS FOR TESTING DATASET BEFORE PRUNING FOR VARIANCE HEURISTIC ******")
print("Number of instances: {}".format(str(testingDataSet.shape[0])))
print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, decisionTreeD))))

print("\nDISTINCT COMBINATIONS OF L AND K FOR VARIANCE HEURISTIC:")
for integerI in range(1, integerL + 1):                    
    pruneTreeDdash = Tree()
    pruneTreeDdash = copy.deepcopy(bestTreeDbest)
    integerM = random.randint(1,integerK)
    pruneTreeDdash = pruningFunction(integerM,pruneTreeDdash)
    accuracyOfNewlyPrunedTree = accuracyCalculator(validationDataSet, pruneTreeDdash)
    print("L: {}, K: {}, Accuracy: {}".format(integerI, integerM, accuracyOfNewlyPrunedTree))
    if accuracyOfNewlyPrunedTree > maximumAccuracy:             # print("Accuracy Improved")
        valueOfLforBestTree = integerI
        valueOfKforBestTree = integerM
        maximumAccuracy = accuracyOfNewlyPrunedTree
        bestTreeDbest = copy.deepcopy(pruneTreeDdash)

print("\n****** STATS FOR TESTING DATASET AFTER PRUNING FOR VARIANCE HEURISTIC ******")
print("Number of instances: {}".format(str(testingDataSet.shape[0])))
print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, bestTreeDbest))))


#print("\n****** STATS FOR ALL DATASETS BEFORE PRUNING FOR VARIANCE HEURISTIC ******")
#print("\n1. TRAINING DATASET")
#print("Number of instances: {}".format(str(trainingDataSet.shape[0])))
##print("Number of attributes: {}".format(str(trainingDataSet.shape[1] - 1)))
#print("Accuracy: {}%".format(str(accuracyCalculator(trainingDataSet, decisionTreeD))))
#
#print("\n2. VALIDATION DATASET")
#print("Number of instances: {}".format(str(validationDataSet.shape[0])))
#print("Accuracy: {}%".format(str(accuracyCalculator(validationDataSet, decisionTreeD))))
#
#print("\n3. TESTING DATASET")
#print("Number of instances: {}".format(str(testingDataSet.shape[0])))
#print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, decisionTreeD))))

if toPrint == 'yes':
    print("\n******PRINTING DECISION TREE AFTER PRUNING (VARIANCE HEURISTIC) *******")
    bestTreeDbest.printTree(bestTreeDbest.root)
    print("********* FINISHED PRINTING DECISION TREE BASED ON VARIANCE HEURISTIC ************")    

#print("\n******STATS FOR ALL DATASETS AFTER PRUNING FOR VARIANCE HEURISTIC******")
#print("\n1. TRAINING DATASET")
#print("Number of instances: {}".format(str(trainingDataSet.shape[0])))
#print("Accuracy: {}%".format(str(accuracyCalculator(trainingDataSet, bestTreeDbest))))
#
#print("\n2. VALIDATION DATASET")
#print("Number of Instances: {}".format(str(validationDataSet.shape[0])))
#print("Accuracy: {}%".format(str(accuracyCalculator(validationDataSet, bestTreeDbest))))
#
#print("\n3. TEST DATASET")
#print("Number of instances: {}".format(str(testingDataSet.shape[0])))
#print("Accuracy: {}%".format(str(accuracyCalculator(testingDataSet, bestTreeDbest))))
#
#print("\n")
#
#accuracyOnValidationSetUnpruned = accuracyCalculator(validationDataSet, decisionTreeD)
#if(maximumAccuracy > accuracyOnValidationSetUnpruned):
#    print("Pruned successfully when L: {}, K: {}. Total pruning attempts: {}".format(valueOfLforBestTree, valueOfKforBestTree, integerL))
#    print("Pre-pruning accuracy: {}%; Post-pruning accuracy: {}%".format(accuracyOnValidationSetUnpruned, maximumAccuracy))
#else:
#    print("Decision Tree pruned. However, accuracy didn't improve after {} attempts, given by value of integer L.".format(integerL))

print("\n********** End of program **********")
