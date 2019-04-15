import numpy as np

def accuracy_score(ytrue, ypred):
    n_correct = 0
    n_total = 0
    for i, pred in enumerate(ypred):
        try:
            if pred == ytrue[i]:
                n_correct += 1
        except:
            if np.argmax(pred) == np.argmax(ytrue[i]):
                n_correct += 1
        n_total += 1
    ratio = n_correct / n_total
    accuracy = ratio * 100
    return round(accuracy, 2)