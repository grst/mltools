import re
from .mltools import randsample

def extract_confusion_matrix(wekafile):
    matrix = []
    with open(wekafile) as f:
        isMatrix = False
        isCV = False
        for line in f.readlines():
            if line.startswith("=== Stratified cross-validation"):
                isCV = True
            if isCV and line.startswith("=== Confusion Matrix"):
                isMatrix = True
            if isMatrix and isCV:
                matrix.append(line)
    pos = matrix[3].split("|")[0].strip()
    neg = matrix[4].split("|")[0].strip()
    TP, FN = (int(x.strip()) for x in pos.split())
    FP, TN = (int(x.strip()) for x in neg.split())
    return (TP, FN, FP, TN)

def extract_confusion_matrix_instances(wekafile):
    """
    return list [(actual, predicted), (actual, predicted), ...]
    """
    dataLine = re.compile(r"\d+")
    instances = []
    with open(wekafile) as f:
        for line in f.readlines():
            if dataLine.search(line):
                actual, predicted = line.strip().split()[1:3]
                actual = actual[2:]
                predicted = predicted[2:]
                instances.append((actual, predicted))
    return instances

def get_confusion_matrix(instances):
    TP = FN = FP = TN = 0
    for actual, predicted in instances:
        if actual == "positive":
            if predicted == "positive":
                TP += 1
            elif predicted == "negative":
                FN += 1
            else:
                assert False, "invalid predicted class"
        elif actual == "negative":
            if predicted == "positive":
                FP += 1
            elif predicted == "negative":
                TN += 1
            else:
                assert False, "invalid predicted class"
        else:
            assert False, "invalid actual class"
    return (TP, FN, FP, TN)

def bootstrap(instances, iterations=1000, fraction=0.5):
    perfs = []
    for i in range(iterations):
        perfs.append(get_confusion_matrix(randsample(instances)))
    return perfs
