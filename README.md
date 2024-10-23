# Data Science Internship at Prodigy Infotech - Task Summary

This repository contains the projects completed during my 1-month Data Science internship at Prodigy Infotech. The internship involved working on various data science tasks ranging from data visualization and exploratory data analysis to machine learning and sentiment analysis.

## Table of Contents
1. [Task 1: Data Visualization](#task-1-data-visualization)
2. [Task 2: Data Cleaning and Exploratory Data Analysis (EDA)](#task-2-data-cleaning-and-exploratory-data-analysis-eda)
3. [Task 3: Decision Tree Classifier](#task-3-decision-tree-classifier)
4. [Task 4: Sentiment Analysis on Social Media Data](#task-4-sentiment-analysis-on-social-media-data)
5. [Task 5: Traffic Accident Data Analysis](#task-5-traffic-accident-data-analysis)

## Task 1: Data Visualization
**Objective:** Create a bar chart or histogram to visualize the distribution of a categorical or continuous variable.

- Dataset: A sample population dataset (e.g., ages, genders).
- Tools: Python, Matplotlib, Seaborn.
- Description: Visualized the distribution of ages/genders using bar charts and histograms.

### Code Example
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data
data = [20, 21, 22, 23, 24, 25, 26, 22, 20, 21, 23, 25]
plt.hist(data, bins=5)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

## Task 2: Data Cleaning and Exploratory Data Analysis (EDA)
**Objective:** Perform data cleaning and EDA on a dataset to explore the relationships between variables and identify patterns/trends.

-Dataset: Titanic dataset from Kaggle.
-Tools: Python, Pandas, Matplotlib, Seaborn.
-Description: Cleaned missing data, performed statistical analysis, and visualized key relationships.

### Code Example
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic Dataset
titanic_data = pd.read_csv('titanic.csv')

# Cleaning missing values
titanic_data.fillna(method='ffill', inplace=True)

# Exploratory Data Analysis
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()
```
## Task 3: Decision Tree Classifier
**Objective:** Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

-Dataset: Bank Marketing dataset from UCI Machine Learning Repository.
-Tools: Python, Scikit-learn, Pandas.
-Description: Trained a decision tree classifier to predict customer behavior.

### Code Example
```python

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
bank_data = pd.read_csv('bank_marketing.csv')

# Preprocess data
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

## Task 4: Sentiment Analysis on Social Media Data
**Objective:** Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes.

-Dataset: Social media comments related to specific topics/brands.
-Tools: Python, NLTK, TextBlob, Matplotlib.
-Description: Performed sentiment analysis on social media data, visualized positive, neutral, and negative sentiments.

### Code Example
```python
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample Social Media Comments
comments = ["I love this product!", "This is the worst service ever.", "Quite satisfactory overall."]

# Sentiment Analysis
sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]

# Visualization
plt.bar(range(len(comments)), sentiments)
plt.title('Sentiment Analysis of Social Media Comments')
plt.xlabel('Comment')
plt.ylabel('Sentiment Polarity')
plt.show()
```
## Task 5: Traffic Accident Data Analysis
**Objective:** Analyze traffic accident data to identify patterns related to road conditions, weather, and time of day. Visualize accident hotspots and contributing factors.

-Dataset: UK Road Safety Dataset.
-Tools: Python, Pandas, Matplotlib, Seaborn.
-Description: Identified key patterns in accident data, such as high-risk road conditions and accident hotspots.
### Code Example
```python

# Load Traffic Accident Data
accident_data = pd.read_csv('Accident_Information.csv')

# Visualize Accident Hotspots by Location
sns.scatterplot(x='Longitude', y='Latitude', data=accident_data)
plt.title('Accident Hotspots')
plt.show()

# Analyze the effect of weather on accidents
sns.countplot(x='Weather_Conditions', data=accident_data)
plt.title('Accidents by Weather Condition')
plt.show()
```
## Conclusion
Throughout the internship, I gained hands-on experience in data analysis, data visualization, machine learning, and sentiment analysis using real-world datasets. Each task provided valuable insights into different aspects of data science, from data cleaning to building predictive models.
