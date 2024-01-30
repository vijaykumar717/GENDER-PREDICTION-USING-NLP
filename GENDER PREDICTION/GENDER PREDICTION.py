from nltk.corpus import names
import random
def gender_features(word):
    return {'lastletter' : word[-1]}
import nltk
from nltk.corpus import names
from nltk import NaiveBayesClassifier as NBC
from nltk import classify
import random
nltk.download('names')
maleNames = [ (name, 'male') for name in names.words('male.txt') ]
femaleNames = [ (name, 'female') for name in names.words('female.txt') ]
allNames = maleNames + femaleNames
random.shuffle(allNames)
featureData = [ (gender_features(namelist), gender) for (namelist, gender) in allNames ]
test_data = featureData[:700]
train_data = featureData[700:]
classifier = NBC.train(train_data)
print(classifier.classify(gender_features('prabha')))
