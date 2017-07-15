import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])
# function that transforms the strings to binary data
def toBinary(label):
   if(label=="ham"):
     return 0
   else:
     return 1

# Apply the function to our dataset
df['label'] = map(toBinary,df['label'])

# Training et testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)
# Applying BoW processing to the training set
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)

testing_data = count_vector.transform(X_test)

# Naive Bayes implementation
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

# Measure Accuracy, Precision and Recall(Sensisivity)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
