#Conversion of Dataset into Vector Embeddings

import pandas as pd 
#pandas for csv handling

import numpy as np
#numpy for matrix handling  

from bs4 import BeautifulSoup
#BeautifulSoup for HTML raw text extraction
#Text contains HTML tag <br /> which needs to be replaced with somthing transformer can handle, \n works.

from sentence_transformers import SentenceTransformer
#SentenceTransformer for embedding generation
#Pulled from hugging face sentance transformer libary

from sklearn.neighbors import KNeighborsClassifier
#KNeighborsClassifier for KNN model

from sklearn import svm
#SVM for SVM model

from sklearn.metrics import confusion_matrix, accuracy_score
#sklearn for confusion matrix and accuracy score

import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib and seaborn for confusion matrix visualization

#clean function
def clean_Review(html_text):
    soup = BeautifulSoup(html_text, 'html.parser') #Hand HTML string to BeautifulSoup
    result = soup.get_text('\n') #Extract only the text from HTML, replacing any HTML fragments with \n
    soup.decompose()  # Destroy the soup object
    return result #Retrun the cleaned text

IDMBDataframe = pd.read_csv('.\IMDB_Dataset.csv', 
                            names = ['Reviews', 'Classification'], 
                            skiprows=1)
# Import IDMB Dataset from CSV file into a pandas dataframe for manipulation and labeling
# Dataframe created here is result of .read_csv function

reviewsExclusive = IDMBDataframe.iloc[:, 0]
classificationExclusive = IDMBDataframe.iloc[:, 1]
# Select only the 1st/2nd of the dataframe, which contains the reviews/classification
#.iloc function used to specify row and column index for dataframe

for Index, Review in enumerate(reviewsExclusive):
    cleanReview = clean_Review(Review) #Clean the review using the clean_Review function
    reviewsExclusive.at[Index] = cleanReview #Replace the original review with the cleaned review
    #enumerate function adds a counter to the iterable loop placed into the Index Variable
    #Review holds the string value of the current review being processed

MiniLMEM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#Call model from hugging face sentence transformer library

embeddings = MiniLMEM.encode(reviewsExclusive)
#Encode the cleaned reviews using the MiniLM model

vectorArrFormat = np.array([np.array(item) for item in embeddings])
#Conversion of Dataset into Vector Embeddings

#User input for review to classify
print("Please Enter Movie Review to Classify:")
user_Review = input()

#User input for review to classify
print("Enter The Movie Review Sentiment To Your Judgement {positive/negative}:")
review_Sentiment = input()

#Convert user input into vector format for KNN/SVM model
userVec = MiniLMEM.encode(user_Review)
userArr = np.array(userVec).reshape(1, -1)
sentimentArr = np.array(review_Sentiment).reshape(1, -1)   

# Initialize and train KNN (k=5 as basic example)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(vectorArrFormat, classificationExclusive)

# Make predictions
KNNuserPred = knn.predict(userArr)

print("Model Evaluation:\n-------------------")
print(f"KNN Predicted Sentiment: {KNNuserPred[0]}")

#SVM model Classifier
svmBase = svm.SVC()
svmBase.fit(vectorArrFormat, classificationExclusive)

#Make predictions
SVMuserPred = svmBase.predict(userArr)

print(f"SVM Predicted Sentiment: {SVMuserPred[0]}\n")

if KNNuserPred[0] == sentimentArr[0]:
    print("KNN Model Prediction is Correct!")
else:
    print("KNN Model Prediction is Incorrect!")

if SVMuserPred[0] == sentimentArr[0]:
    print("SVM Model Prediction is Correct!")
else:   
    print("SVM Model Prediction is Incorrect!")

