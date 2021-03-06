# python-Decision-Tree-for-binary-features-with-pruning-with-pandas

![Alt text](screenshot1.PNG?raw=true "Machine-Learning-python-Decision-Tree-for-binary-features-with-pruning-with-pandas")
![Alt text](screenshot2.PNG?raw=true "Machine-Learning-python-Decision-Tree-for-binary-features-with-pruning-with-pandas")
![Alt text](screenshot3.PNG?raw=true "Machine-Learning-python-Decision-Tree-for-binary-features-with-pruning-with-pandas")

This script takes 6 inputs from command line:
1. integerL: number of times pruning will take place
2. integerK: integer used to denote which node will be removed while pruning
3. path-to-training-set: path to training dataset CSV file, containing list of data points with binary features, and classification label 'Class'
4. path-to-validation-set: path to validation dataset CSV file, used for pruning
5. path-to-test-set: path to test dataset, on which accuracy is calculated
6. to-print: whether to print the decision trees or not

The script builds 4 decision trees:
1. 1 based on entropy before pruning.
2. 1 based on entropy after pruning.
3. 1 based on variance before pruning.
4. 1 based on variance after pruning.

VERSION OF PYTHON USED FOR DEVELOPMENT: 3.6

PACKAGE REQUIREMENTS:
The script requires pandas to run. To install pandas from the command line, execute:
$ pip install pandas
This will also instal pytz, six, python-dateutil, numpy, if not already installed on system, in addition to pandas.

TO RUN THE SCRIPT:
1. Open command prompt/terminal.
2. Navigate to the directory containing the script.
3. Execute the following command:
$ python script.py <L> <K> <path-to-training-set> <path-to-validation-set> <path-to-test-set> <to-print>
examples: 
$ python script.py 10 10 "D:\ML\Decision Tree\data_sets1\training_set.csv" "D:\ML\Decision Tree\data_sets1\validation_set.csv" "D:\ML\Decision Tree\data_sets1\test_set.csv" "no"
$ python ascript.py "10" "10" "D:\ML\Decision Tree\data_sets1\training_set.csv" "D:\ML\Decision Tree\data_sets1\validation_set.csv" "D:\ML\Decision Tree\data_sets1\test_set.csv" "yes"

I have used raw strings to parse the first three paths, so single backward slashes should work. If the dataset CSVs are placed in the same directory as the python file, just mention the file names. Example:
$ python script.py "10" "10" "training_set.csv" "validation_set" "test_set.csv.csv" "yes"

OUTPUT:
When toPrint == 'yes', the script produces the following outputs:
a. Decision tree based on Entropy heuristic before pruning.
b. Accuracy on test dataset for Entropy heuristic before pruning.
c. Accuracies for post-pruned versions of the tree for Entropy heuristic for given values of L and K.
d. Accuracy on test dataset for Entropy heuristic after pruning.
e. Best decision tree based on Entropy heuristic after L attempts.
f. Accuracy on test dataset for Variance heuristic before pruning.
g. Accuracies for post-pruned versions of the tree for Variance heuristic for given values of L and K.
h. Decision tree based on Variance heuristic before pruning.
i. Accuracy on test dataset for Variance heuristic after pruning.
j. Best decision tree based on Variance heuristic after L attempts.

When toPrint != 'yes', the script doesn't print the decision trees. It prints the following outputs only:
a. Accuracy on test dataset for Entropy heuristic before pruning.
b. Accuracies for post-pruned versions of the tree for Entropy heuristic for given values of L and K.
c. Accuracy on test dataset for Entropy heuristic after pruning.
d. Accuracy on test dataset for Variance heuristic before pruning.
e. Accuracies for post-pruned versions of the tree for Variance heuristic for given values of L and K.
f. Accuracy on test dataset for Variance heuristic after pruning.
