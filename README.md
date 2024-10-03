# Classification-of-the-category-using-machine-learning-on-BBC-news-dataset

This work is a news stream categorization system with features that employs machine learning approach for categorizing news articles into different categories including Business, Tech, Politics, Sports, and Entertainment. The data used in this work is the BBC News Dataset and several models are applied to classify and predict the category of news articles.

# Installation
To manage this project, you should have Python or any other compatible later on and the following libraries. You can install the required libraries using pip:
## pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn tensorflow
Make sure to also download the necessary NLTK data files by running:
## import nltk
## nltk.download('stopwords')
## nltk.download('wordnet')
## nltk.download('punkt')


## Data Exploration
The dataset is made up of many news articles accompanied by their categories. The initial exploration includes:
Exploiting the initial elements of accompanying the series on the display.
Both the shape and the information of the displayed dataset.
The author carries out a visualization of different categories using bar plots and pie charts.

## Text Preprocessing
Preprocessing is a very important part of natural language processing. This project employs several preprocessing techniques, including:
Stripping of HTML tags as well as special characters.
Converting text to lowercase.
Removing stop words.
Stemming of words to bring them to their stem.

## Feature Extraction
For feature extraction, the project uses:
CountVectorizer: Takes the collection of text documents and transforms it into an opposite matrix of token frequency.
The features are finally restricted to those with the 5000 highest frequency counts out of all words encountered in the corpus.

## Model Building
The following machine learning models are built and evaluated:
Logistic Regression
Random Forest Classifier
Multinomial Naive Bayes
SVC which stands for Support Vector Classifier
Decision Tree Classifier
TensorFlow Neural Network (for multi class classification)
The splits of data into the training and testing set are used to train and test each model.

## Model Evaluation
The models are evaluated based on the following metrics:
Accuracy
Precision
Recall
F1-score
Performance results are saved and hence it is used enable comparison in a bid to determine the best performing model.

## Results
At the end of the model performance the accuracy and several other metrics are printed in a table format for all classifiers. It is also possible to make the new articles’ predictions by transforming the articles and using the developed models.

## Example of Predictions
An example of the text predicting the category of a news article is given to illustrate an example of using the TextBlob for the classification of new text input into the desired category.

## License
As with most projects, this one is also licensed through the MIT License. It is okay to change and enable the code as the case maybe in your class, institution or project.
