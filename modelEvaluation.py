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

from sklearn.model_selection import train_test_split
#sklearn for train/test split

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

IDMBDataframe = pd.read_csv('.\IMDB Dataset.csv', 
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

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    vectorArrFormat, classificationExclusive, 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Initialize and train KNN (k=5 as basic example)
print("\nTraining KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nKNN Model accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nKNN Confusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#SVM model Classifier
print("\nTraining SVM model...")
svmBase = svm.SVC()
svmBase.fit(X_train, y_train)

#Make predictions
svmPredictions = svmBase.predict(X_test)

#score the predictions
svmBase.score(X_test, y_test)
print(f"\nSVM Model accuracy: {svmBase.score(X_test, y_test):.4f}")

#generate confusion matrix
SVMcMat = confusion_matrix(y_test, svmPredictions)
print("\nSVM Confusion Matrix:")
print(SVMcMat)

#Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(SVMcMat, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
