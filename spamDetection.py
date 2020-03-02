
""" Phase - I. Import Libraries """

import sys
import nltk
import sklearn
import pandas
import numpy

print("Python: {}".format(sys.version))
print("NLTK: {}".format(nltk.__version__))
print("Pandas: {}".format(pandas.__version__))
print("sklearn: {}".format(sklearn.__version__))
print("Numpy: {}".format(numpy.__version__))



""" Phase - II. Load Dataset in Pandas Dataframe """

import pandas as pd 
import numpy as np 
import re


# Load the dataset of SMS messages or download data from "https://archive.ics.uci.edu/ml/datasets/sms+spam+collection"

dataframe = pd.read_table('/home/shaileshsarda/Desktop/NLP Project/smsspamcollection/SMSSpamCollection', header = None, encoding='utf-8')
print(dataframe.head())  # View 1st 5 rows



# Get the info about attributes
print(dataframe.info())


# To check the target value means 0 and seperate ham and spam entries
# This shows the the ham and spam contains how many entries 
distinct_classes = dataframe[0]
print(distinct_classes.value_counts())



""" Phase - III. Perform some Data Preprocessing """

# Convert distinct_classes labels to binary values, 0 = ham and 1 = spam
from sklearn.preprocessing import LabelEncoder

classes_encoder = LabelEncoder()
classes_encoder_output = classes_encoder.fit_transform(distinct_classes)

print(classes_encoder_output[:10])



# Store the SMS message data seperately
messages = dataframe[1]
print(messages[:10])


# Apply text preprocessing on this messages parameters to remove the special characters, any patterns like emails, urls, phoneNumber etc


# 1. Replace email address from messages attribute by 'emailaddress'

processed_data = messages.str.replace(r"[^@]+@[^@]+\.[^@]+",'emailaddress')


# 2. Replace Website URL from messages attribute by 'webaddress'

processed_data = processed_data.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

# 3. Replace Currency Symbols from messages attribute by 'currencysymbl'

processed_data = processed_data.str.replace(r"£|\$",'currencysymbl')

# 4. Replace phone number from messages attribute by 'phonenumber'

processed_data = processed_data.str.replace(r"^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$",'phonenumber')


# 5. Replace numbers from messages attribute by 'number'

processed_data = processed_data.str.replace(r"\d+(\.\d+)?",'number')


# 6. Remove the whitespaces

processed_data = processed_data.str.replace(r'\s+', ' ')


# 7. Remove the Punctuation

processed_data = processed_data.str.replace(r'[^\w\d\s]', ' ')

# 8. Remove leading and trailing whitespace

processed_data = processed_data.str.replace(r'^\s+|\s+?$', '')


# 9. Convert in lower case

processed_data = processed_data.str.lower()
print(processed_data)




"""
##### Or We can write a simple function in pythonn ##########
def preprocess_data(messages):
	processed_data = messages.str.replace(r"[^@]+@[^@]+\.[^@]+",'emailaddress')
	processed_data = processed_data.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
	processed_data = processed_data.str.replace(r"£|\$",'currencysymbl')
	processed_data = processed_data.str.replace(r"^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$",'phonenumber')
	processed_data = processed_data.str.replace(r"\d+(\.\d+)?",'number')
	processed_data = processed_data.str.replace(r'\s+', ' ')
	processed_data = processed_data.str.replace(r'[^\w\d\s]', ' ')
	processed_data = processed_data.str.replace(r'^\s+|\s+?$', '')
	processed_data = processed_data.str.lower()
	print(preprocess_data)
	return preprocess_data


preprocess_data(messages)


"""



""" Phase - IV.  To start actual Natural Language Processing from here """

# 1.  Remove stopwords from contents

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

processed_data = processed_data.apply(lambda x: ' '.join(i for i in x.split() if i not in stop_words))



# 2.  Applying stemming 

ps = nltk.PorterStemmer()

processed_data = processed_data.apply(lambda x: ' '.join(ps.stem(item) for item in x.split()))






""" Phase V. Identify Features : Here the words in the messages will become the features."""

# 1.  create bag-of-words

from nltk.tokenize import word_tokenize

list_words = []

for message in processed_data:
    words = word_tokenize(message)
    for w in words:
        list_words.append(w)
        
all_words = nltk.FreqDist(list_words)



# 2.  print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(list_words)))




# 3. We will use the 1800 most common words as features
word_features = list(list_words)[:1800]


# 4. The find_features function will determine which of the 1800 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# 5.  Lets see an example!
features = find_features(processed_data[0])
for key, value in features.items():
    if value == True:
        print(key)




# 6. Now lets do it for all the messages
messages = list(zip(processed_data, classes_encoder_output))



# 7. define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)


# 8. call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]


# 9. Now split the featuresets into training and testing datasets using sklearn
from sklearn.model_selection import train_test_split

# 10. split the data into training and testing datasets
training, testing = train_test_split(featuresets, test_size = 0.30, random_state=seed)


# 11. Check the length of the data 
print(len(training))
print(len(testing))



""" Phase VI. Apply Scikit-Learn Classifiers with NLTK """

# 1. Support Vector Machine
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

support_vector_model = SklearnClassifier(SVC(kernel = 'linear'))


# train the model on the training data
support_vector_model.train(training)

# Test model on testing data set

accuracy = nltk.classify.accuracy(support_vector_model, testing)*100 # Answer : 98.68421052631578 %
print("SVM Model Accuracy: {}".format(accuracy),"%")


# 2. Define other models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Define models to train
model_names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]


classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]



models = zip(model_names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))



#  Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))

























