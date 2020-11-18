from sklearn import svm
import numpy as np
from joblib import dump, load

def svm_learn(DataFrame):
    lenth = len(DataFrame)
    np.random.shuffle(DataFrame)
    Xtrain = DataFrame[:lenth-40, :-1]
    Ytrain = DataFrame[:lenth-40, -1]
    Xeval = DataFrame[lenth-40:, :-1]
    Yeval = DataFrame[lenth-40:, -1]
    clf = svm.SVC(gamma = 'auto')
    clf.fit(Xtrain, Ytrain)
    print('Evaluating model....')
    print(str(len(Xeval)), ' elements')
    truePred = 0
    for i in range(len(Xeval)):
        e = clf.predict([Xeval[i]])
        print(str(i), 'SVM model predicted: ', str(e))
        print('  True valeu was: ', str(Yeval[i]))
        if e == Yeval[i]:
            truePred += 1
    print('Accuracy of the model: ', str(truePred/len(Xeval)))
    dump(clf, 'models\yawing_classifier.joblib') 