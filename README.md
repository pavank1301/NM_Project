# NM_Project
# AI Powered Spam Classifier
IBM-NM-Project
Domain: AI
Title: Building a Smarter AI-Powered Spam Classifier
Presenter: Pavan Kumar V (513521104034)
College: ANNAI MIRA COLLEGE OF ENGINEERING AND TECHNOLOGY

INRODUCTION:
•	     In today's digital landscape, the challenge of distinguishing genuine messages from spam has become more critical than ever. The relentless barrage of unsolicited and potentially harmful content poses a significant inconvenience and security risk to users. Accurate spam classification is the key to separating the wheat from the chaff, helping users make informed decisions about their incoming messages.
     
PROJECT OVERVIEW:
This project's primary objective is to distinguish between genuine and spam messages in digital communication. We employ Natural Language Processing (NLP) techniques to build an effective spam classification model. NLP enables the model to analyze message content for patterns and characteristics. The collaborative team effort ensures a comprehensive approach to spam classification, enhancing communication security and user experience.

DESIGN THINKING:
Our approach to solving this problem can be classified into several phases, each with specific objectives and tasks. 
This structured approach will ensure that we systematically address all aspects of the problem.
These phases will include:
1.	Acquiring Data from Data Source
2.	Data preprocessing
3.	Feature Engineering
4.	Text Preprocessing
5.	Feature Extraction
6.	Model Selection
7.	Model Building
8.	Model Fitting 
9.	Model Training
10.	Evaluation 

LIST OF TOOLS AND SOFTWARE COMMONLY USED IN THE PROCESS:
1.	Programming Language - Python
2.	Integrated Development Environment(IDE) - Jupyter Notebook
3.	Machine Learning Libraries - scikit-learn, Tensorflow
4.	Data Visualization Tools: matplotlib, seaborn, 
5.	Data Preprocessing Tools: pandas library,  nltk library

Step 1- ACQUIRING DATA:
We have obtained the datasets from “Kaggle” that contains the list of articles that are considered to be true and fake. It helps in the distinction between the fake news from the true news.
The link for the dataset is : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
Step 2- DATA PRPEPROCESSING:
Labeling: If your dataset does not already have labels, you should assign labels to each data point. In the case of a spam classifier, you can use "spam" and "ham" (non-spam) labels to categorize the messages.

Handling Null Values: Check for any null or missing values in your dataset. Depending on the nature and quantity of missing data, you can either remove rows with missing values or fill them with appropriate values.

Lowercasing: Convert all text to lowercase to ensure uniformity and prevent the model from treating "Spam" and "spam" as different words.

Removing Special Characters: Remove any special characters, punctuation, and symbols that are not relevant for spam classification. This step helps to reduce noise in the text data.

Stopword Removal: Remove common stopwords (e.g., "the," "and," "in") that do not carry much information for spam detection. NLTK or spaCy libraries provide predefined lists of stopwords that you can use for this purpose.
Step 3- FEATURE ENGINEERING:
Feature Engineering is the process of taking raw data and transforming them into certain features that help in creating a predictive model using standard modelling methods.
It is also a form of Exploratory Data Analysis.
step 4- TEXT PREPROCESSING:
In the context of developing an LSTM-based model for a spam classifier, text preprocessing, including stemming, is essential. Here's how you can perform stemming as part of text preprocessing for your spam classifier:

Tokenization: Tokenize the text data into individual words or tokens. This step involves splitting the text into meaningful units, such as words.

Stemming: Apply stemming to reduce words to their root or base form. Stemming helps in simplifying the text data, making it easier for the model to identify common patterns. There are different stemming algorithms available, and one common one is the Porter Stemmer.

Stopword Removal: As previously mentioned, remove common stopwords as they do not carry significant information for spam detection.
Step 5- FEATURE EXTRACTION:
5.1) Word Embedding by TF-IDF:

Use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert the text data into numerical features. It represents the importance of words in each document relative to the entire corpus.
5.2) One-Hot Representation:

Convert categorical variables into a binary (0/1) format. For text classification tasks, this is not typically used for the text itself but for other categorical features if they exist.
5.3) Padding:

Padding is commonly used for sequences in NLP tasks, such as text classification using recurrent neural networks like LSTM. It ensures that all input sequences are of the same length.
Step 6- MODEL SELECTION:
We should select a suitable classification algorithm (e.g., Logistic Regression, Random Forest, or Naïve bayes, SVM) for the Spam Classifier task.
Among the classification algorithms, we have chosen the LSTM (Long Short Term Memory) for building the model.
LSTM (Long Short Term Memory) which helps in containing sequence information.
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. 

Step 7- MODEL BUILDING:
At first we are going to develop the base model and compile it. 
First layer will be the embedding layer which has the input of vocabulary size, vector features and sentence length. 
Later we add 30% dropout layer to prevent over-fitting and the LSTM layer which has100 neurons in the layer.
In final layer we use sigmoid activation function. Later we compile the model using adam optimizer and binary cross entropy as loss function since we have only two outputs.

Step 8- MODEL FITTING:
Before fitting to the model, we considered the padded embedded object as X and y as y itself and have converted them into an array.

Step 9- MODEL TRAIINING:
We have split our new X and y variable into train and test and proceed with fitting the model to the data. We have considered 10 epochs and 128 as batch size. 
The number of epochs and batch size can be varied to get better results.

Step 10- MODEL EVALAUTION:
Evaluation metrics are used to measure the quality of the statistical or machine learning model. 
Evaluating machine learning models or algorithms is essential for any project. 
There are many different types of evaluation metrics available to test a model. 
Here we have used:
10.1)	Confusion Matrix - It is a table that is used in classification problems to assess where errors in the model were made.
10.2)	Accuracy - Accuracy measures how often the model is correct.
10.3)     Precision - Precision measures the proportion of true positive predictions (spam) among all positive predictions made by the classifier.
CONCLUSION:
In the realm of email and message filtering, the development of a spam classifier using machine learning and natural language processing techniques is a significant step toward enhancing the quality of digital communication and information security. The project's objective was to create a robust model capable of distinguishing between spam and non-spam messages effectively. Through the systematic approach and utilization of various tools, data preprocessing, and model selection, we have achieved notable results and insights.
