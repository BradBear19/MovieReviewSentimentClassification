# Movie Review Sentiment Classification
-------------------------------------
By: Bradford King, Kenneth Howard, Jaden Perry, Gia Huy Phan, Malachi Platt, and Vontrell Jefferson
--------------------------------------------------------------------

SUMMARY::

- Movie Review Sentiment Classification based on Stanford University Large Movie Review Dataset. 

- Two separate programs are included in the zip file: 
- modelEvaluation.py which utilizes an 80%/20% split of the training data to evaluate KNN and SVM model accuracy.
- reviewClassifier.py which takes user string input and utilizes the trained models to evaluate user review.

- Additionally, the zip contains IDMB_Dataset, which acts as the foundation for model training and evaluation.

USER GUIDE
----------

Download & Install:
-------------------
Requirements:
-------------
Python 3.13.3

Libraries:
----------
pandas
numpy
beautifulsoup4
sentence-transformers
scikit-learn
matplotlib
seaborn

Dataset:
--------
Download from the link at the bottom of README and place IDMB Dataset.csv in the same path as modelEvaluation.py and reviewClassifier.py

Execution:
----------
Execute Model Testing Metric:
-----------------------------
1. Execute modelEvaluation.py



Unique String/Movie Review Evaluation:
---------------------------------------
1. Execute reviewClassifier.py

2. When prompted, either manually enter or copy and paste the review to classify.

3. When prompted, enter if you believe the  movie review is:
- positive
- negative

4. The model will evaluate the input movie review, and the program will return results.


TRAINING DATASET
----------------
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data






