import NaiveBayesClassifier as nbc
import numpy as np

def NBTest():
#     print 'Testing Gaussian Naive Bayes'
#     my_gnb = nbc.GaussianNB()
#     """ train set gives height(ft), weight(lbs), foot size(in)"""
#     train_set = np.array([[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10], [5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150,9] ])
#     train_label = np.array(['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female'])
#     my_gnb.train(train_set, train_label)
#     
#     test_set = np.array([6, 130, 8])
#     print my_gnb.predict(test_set)
#     print 'Done'
    
    print 'Testing Text Classifier'
    classes = np.array(['China', 'Not China'])
    train_set = np.array(['chinese beijing chinese', 'chinese chinese shanghai', 'chinese macao', 'tokyo japan chinese'])
    train_label = np.array(['China', 'China', 'China', 'Not China'])
    my_mnb = nbc.MultinomialNBTextClassifier(classes)
    my_mnb.train(train_set, train_label)
    
    test_set = np.array(['chinese chinese chinese tokyo japan'])
    print my_mnb.predict_doc(test_set)
    print 'Done'