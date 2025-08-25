# %% [markdown] cell 1
# # Introduction
# %% [markdown] cell 2
# The dataset provided by Invistico Airlines contains valuable information about their customers' experiences and satisfaction levels. With the aim of predicting future customer satisfaction and improving service quality, this dataset encompasses various customer attributes and feedback on different aspects of their flights. The primary objectives of this dataset are to predict customer satisfaction and identify areas for service improvement.
# %% [markdown] cell 3
# ### Understanding the variables
# %% [markdown] cell 4
# 1. satisfaction: The overall satisfaction level of the customer. It is a categorical variable with options                          "satisfied" or "dissatisfied".
#
# 2. Gender: The gender of the customer. It is a categorical variable with options "male" or "female".
#
# 3. Customer Type: Whether the customer is a "loyal customer" or a "disloyal customer".
#
# 4. Age: The age of the customer.
#
# 5. Type of Travel: The type of travel, such as "Personal Travel" or "Business travel."Class: The class of travel, such as                      "Eco" or "Business."
#
# 6. Class: The class of travel, such as "Eco" or "Business."
#
# 7. Flight Distance: The distance of the flight.
#
# 8. Seat comfort: Customer rating of seat comfort.
#
# 9. Departure/Arrival time convenient: Customer rating of convenience of departure/arrival times.
#
# 10. Food and drink: Customer rating of food and drink quality.
#
# 11. Gate location: Customer rating of gate location.
#
# 12. Inflight wifi service: Customer rating of inflight Wi-Fi service.
#
# 13. Inflight entertainment: Customer rating of inflight entertainment options.
#
# 14. Online support: Customer rating of online customer support.
#
# 15. Ease of Online booking: Customer rating of ease of online booking.
#
# 16. On-board service: Customer rating of on-board service provided by the airline.
#
# 17. Leg room service: Customer rating of leg room service provided during the flight.
#
# 18. Baggage handling: Customer rating of baggage handling.
#
# 19. Checkin service: Customer rating of check-in service.
#
# 20. Cleanliness: Customer rating of cabin cleanliness.
#
# 21. Online boarding: Customer rating of online boarding process.
#
# 22. Departure Delay in Minutes: The departure delay in minutes for each flight.
#
# 23. Arrival Delay in Minutes: The arrival delay in minutes for each flight.
#
# * All rating features measured on a scale from 0 to 5, where higher values indicate greatersatisfaction.
# %% Code cell 5
# Import libraries. begin, let's import the necessary libraries that we'll be using throughout this notebook:

# Data Manipulation Libraries
import numpy as np 
import pandas as pd 

# Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# %% Code cell 6
# knowing the name of the dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% Code cell 7
# Load tha data.
df = pd.read_excel(r"C:\Users\nithi\Desktop\Invistico_Airline.xlsx")
df.head()

# %% [markdown] cell 8
# # preparation the train data
# %% Code cell 9
# Seeing the shape of the data.
df.shape

# %% Code cell 10
# Seeing if there are dublicated.
df.duplicated().sum()

# %% Code cell 11
# seeing if there are null values.
df.isna().sum()

# %% Code cell 12
# Seeing information about data.
df.info()

# %% Code cell 13
categorical_features = ['satisfaction', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Seat comfort',
                       'Departure/Arrival time convenient', 'Food and drink', 'Gate location',
                       'Inflight wifi service', 'Inflight entertainment', 'Online support',
                       'Ease of Online booking', 'On-board service', 'Leg room service',
                       'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding']

for i in categorical_features:
    print(df[i].value_counts())
    print('-' * 50)

# %% [markdown] cell 14
# In our dataset, consisting of 129,880 rows and 23 columns, we observed no duplicated records, but there are 393 missing values specifically in the 'Arrival Delay in Minutes' column. The data distribution is largely balanced across the various features.
#
# Upon closer examination, we identified a significant issue within certain 'rating' features. These features exhibit extremely low values in rate '0'. This poses a challenge for meaningful analysis.
#
# To address this, we made the decision to combine votes on rate '0' with votes on rate '1'. This step is essential as the majority of our features contain a limited number of 0 ratings compared to other ratings. Moreover, maintaining a rating scale from 1 to 5 is deemed appropriate for our analysis.
# %% Code cell 15
# Define a mapping dictionary to combine the clusters
cluster_mapping = {
    'Seat comfort': {0: 1},
    'Departure/Arrival time convenient': {0: 1},
    'Food and drink': {0: 1},
    'Gate location': {0: 1},
    'Inflight wifi service': {0: 1},
    'Inflight entertainment': {0: 1},
    'Online support': {0: 1},
    'Ease of Online booking': {0: 1},
    'On-board service': {0: 1},
    'Leg room service': {0: 1},
    'Baggage handling': {0: 1},
    'Checkin service': {0: 1},
    'Cleanliness': {0: 1},
    'Online boarding': {0: 1}
}

# Update the specified columns with the new cluster labels using applymap
for col, mapping in cluster_mapping.items():
    df[col] = df[col].apply(lambda x: mapping.get(x, x))

# %% Code cell 16
# Check if the rating features have a rate from 1 to 5 was done correctly
for i in ['Gate location', 'Inflight wifi service', 'Online support', 'Ease of Online booking', 
          'On-board service', 'Leg room service', 'Checkin service', 'Cleanliness', 'Online boarding']:
    print(df[i].value_counts())
    print('-' * 50)

# %% [markdown] cell 17
#
# Following the refinement of the rating features, our next step is to investigate the correlations among the numerical data variables.
# %% Code cell 18
# Numerical columns
numerical_features = ["Age", "Flight Distance", "Departure Delay in Minutes", 'Arrival Delay in Minutes']

# %% Code cell 19
plt.figure(figsize=(8, 6))

# Calculate the correlation matrix for the numerical columns
correlation_matrix = df[numerical_features].corr()

# Create a heatmap using the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Set the title of the heatmap
plt.title('Correlation Heatmap')

# Display the heatmap
plt.show()

# %% [markdown] cell 20
# It is evident that there is an extremely strong correlation of 0.97 between "Departure Delay in Minutes" and 'Arrival Delay in Minutes'. This high correlation suggests that these two variables essentially represent the same information, and therefore, it is advisable to eliminate one of them. Given that 'Arrival Delay in Minutes' also contains missing values, we have decided to proceed with the removal of this variable. This simplification will streamline our analysis while retaining the essential information.

# %% Code cell 21
# Drop the 'Arrival Delay in Minutes' column from the dataframe
df.drop('Arrival Delay in Minutes', axis=1, inplace=True)

# %% [markdown] cell 22
# # Data Visualiation and Analysis
# %% Code cell 23
# List of rating features to be considered for analysis
rating_features = ['Seat comfort', 'Departure/Arrival time convenient', 'Food and drink', 'Gate location', 
                   'Inflight wifi service', 'Inflight entertainment', 'Online support', 'Ease of Online booking',
                   'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness',
                   'Online boarding']

# %% Code cell 24
# Set up subplots with 7 rows and 2 columns
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(16, 16))

# Iterate through the rating features and create bar plots
for i, feature in enumerate(rating_features):
    row, col = divmod(i, 2)
    sns.barplot(x=feature, y='satisfaction', data=df, ax=axes[row, col], palette=['salmon', 'skyblue'])
    
    # Setting titles, x-axis labels, and y-axis labels for each subplot
    axes[row, col].set_title(f'Satisfaction vs {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Satisfaction')

# Adjust subplot layout
plt.tight_layout()

# Show the plots
plt.show()

# %% [markdown] cell 25
# In these graphs, our objective is to discern whether ratings have a direct impact on customer satisfaction or, in other words, to identify the most influential features affecting customer satisfaction. Our analysis reveals that "Departure/Arrival time convenient" and "Gate location" appear to have almost no effect on customer satisfaction. The ratio of satisfied to dissatisfied customers is nearly equal for these two features.
#
# Interestingly, we observe that "Inflight entertainment" plays a pivotal role in customer satisfaction, with a substantial difference between the satisfaction levels of those who are satisfied and dissatisfied. Consequently, it can be considered the most influential feature. Additionally, "Seat comfort," "Online support," "Ease of Online booking," "On-board service,", "Leg room service" and "Online boarding" closely follow "Inflight entertainment" in terms of importance.
#
# However, other features, while still contributing to customer satisfaction, exhibit a comparatively weaker influence.
#
# This analysis assists in prioritizing which aspects of the airline's service may require particular attention for enhancing overall customer satisfaction.
# %% Code cell 26
# Iterate through each rating column
for col in rating_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='satisfaction', palette=['salmon', 'skyblue'])
    
    # Setting the title, labels, and rotation for the x-axis ticks on a plot
    plt.title('Distribution of {}'.format(col))
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Display the plot
    plt.show()

# %% [markdown] cell 27
# In these visualizations, we delve deeper into understanding customer satisfaction in relation to their provided ratings. Notably, for 'Inflight entertainment,' we observe that the majority of customers who give ratings of 4 and 5 are highly satisfied. Remarkably, those who give a rating of 5 are almost exclusively satisfied, indicating an absence of dissatisfied customers in this category. Conversely, those rating 1, 2, or 3 are generally dissatisfied, highlighting that customer ratings for 'Inflight entertainment' distinctly reflect their satisfaction.
#
# Similarly, when examining 'Seat comfort,' 'Online support,' 'Ease of Online booking,' 'On-board service,', 'Leg room service,' and 'Online boarding' we notice that most customers who provide ratings of 4 or 5 express satisfaction, unlike lower ratings. This suggests that these factors are pivotal in determining customer satisfaction.
#
# Conversely, 'Departure/Arrival time convenient' and 'Gate location' do not exhibit a clear pattern in their ratings, as customers giving ratings of "4" 50% of them are satisfied and 50% are not satisfied. This indicates a limited impact of these features on customer satisfaction.
#
# Summarily, the distribution of customer ratings across the various features demonstrates varying degrees of influence on customer satisfaction. 'Inflight entertainment' takes precedence, followed by 'Seat comfort' and 'Online support,' while the remaining features show less pronounced effects.
#
# In conclusion we recommend to:
#
# 1. Enhance Inflight Entertainment: Given its substantial impact on customer satisfaction, investing in and improving inflight entertainment options could yield significant benefits.
#
# 2. Prioritize Seat Comfort: Focus on enhancing seat comfort to further elevate customer satisfaction levels.
#
# 3. Strengthen Online Support: Continue to provide robust online support services, as they contribute positively to customer satisfaction.
#
# 5. Monitor and Improve Other Features: While other features show less pronounced effects, they should not be ignored. Continue to monitor and make incremental improvements to all aspects of the airline service to maintain overall customer satisfaction.
#
# These recommendations are based on our analysis of the correlation between customer ratings and satisfaction levels for each feature and should guide efforts to enhance customer experience and satisfaction.
# %% Code cell 28
# Iterate through each categorical column
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='satisfaction', palette=['salmon', 'skyblue'])
    
    # Setting the title, labels, and legend for a plot depicting customer satisfaction by a specific feature
    plt.title('Customer Satisfaction by {}'.format(col))
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Satisfaction', loc='upper right')
    
    # Display the plot
    plt.show()

# %% [markdown] cell 29
# These visuals highlight several significant trends: firstly, women tend to exhibit higher levels of satisfaction compared to men. Secondly, customers in the 'Business' travel category generally report higher satisfaction levels than those in 'Eco' and 'Eco Plus.' This discrepancy likely arises from the enhanced and more comfortable services typically associated with 'Business' class.
#
# Additionally, the data indicates that 'Business' travelers are consistently more satisfied than 'Personal' travelers, who exhibit a nearly equal ratio of satisfied to dissatisfied customers. Lastly, 'loyal' customers tend to express higher satisfaction levels, while 'disloyal' customers are predominantly dissatisfied.
#
# Our current focus is to uncover the reasons behind the greater satisfaction of 'Business' travelers and the recurring dissatisfaction among 'disloyal' customers. Could the presence or absence of specific features, such as 'Inflight entertainment,' play a role in these patterns?
# %% Code cell 30
plt.figure(figsize=(10, 6))

# Plotting histograms 'Inflight entertainment' ratings by different classes
sns.histplot(df[df['Class'] == 'Business']['Inflight entertainment'], label='Business', kde=True)
sns.histplot(df[df['Class'] == 'Eco']['Inflight entertainment'], label='Eco', kde=True)
sns.histplot(df[df['Class'] == 'Eco Plus']['Inflight entertainment'], label='Eco Plus', kde=True)

# Setting the x-axis label, y-axis label, plot title, and adding a legend
plt.xlabel('Inflight entertainment Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Inflight Entertainment Rating by Class')
plt.legend()

# Displaying the plot
plt.show()

# %% [markdown] cell 31
# This graph illustrates a clear trend: 'Eco' and 'Eco Plus' passengers tend to provide lower ratings for 'Inflight entertainment,' while 'Business' passengers consistently rate it higher. This observation aligns with our previous finding that customers who assign ratings of 5 or 4 to 'Inflight entertainment' are generally satisfied.
#
# The reason for the heightened satisfaction among 'Business' passengers becomes apparent: they enjoy more extensive 'Inflight entertainment' services compared to 'Eco' and 'Eco Plus' passengers. This difference in available entertainment options emerges as a likely explanation for the varying levels of satisfaction.
#
# In conclusion we recommend to:
# 1. Enhance Inflight Entertainment in Economy Classes: To improve customer satisfaction in 'Eco' and 'Eco Plus,' consider enhancing the Inflight entertainment offerings in these classes to align them more closely with the higher-rated 'Business' class.
#
# 2. Highlight Inflight Entertainment as a Feature: In marketing and service promotions, emphasize the availability and quality of Inflight entertainment, especially in 'Business' class, to attract and retain customers who value this feature.
# %% Code cell 32
plt.figure(figsize=(10, 6))

# Plotting histograms 'Inflight entertainment' ratings by different Customer Type
sns.histplot(df[df['Customer Type'] == 'Loyal Customer']['Inflight entertainment'], label='Loyal Customer', kde=True)
sns.histplot(df[df['Customer Type'] == 'disloyal Customer']['Inflight entertainment'], label='disloyal Customer', kde=True)

# Setting the x-axis label, y-axis label, plot title, and adding a legend
plt.xlabel('Inflight entertainment Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Inflight Entertainment Rating by Type of Travel')
plt.legend()

# Displaying the plot
plt.show()

# %% [markdown] cell 33
# This graph also underscores the significance of 'Inflight Entertainment' in shaping customer satisfaction levels. 'Disloyal' customers tend to provide lower ratings, compared to 'loyal' customers. This underscores why 'disloyal' customers often express dissatisfaction and highlights the crucial role of investing in and enhancing this feature.
# %% Code cell 34
plt.figure(figsize=(10, 6))

# Create a violin plot
sns.violinplot(x='satisfaction', y='Age', data=df, palette=['salmon', 'skyblue'])

# Add labels and title
plt.xlabel('Satisfaction')
plt.ylabel('Age')
plt.title('Age vs. Satisfaction')

# Show the plot
plt.show()

# %% [markdown] cell 35
# The plot of the violin clearly demonstrates a marked variance in the age distribution between satisfied and dissatisfied individuals. It reveals that satisfied individuals generally show a higher life expectancy (from about 40 to 60) than their dissatisfied counterparts(ranging from 20 to 40). This observation points to a possible influence of age on levels of satisfaction, as older individuals tend to express greater satisfaction.
#
# One plausible explanation for this trend is that younger individuals often have more specific and demanding expectations, which makes meeting their satisfaction standards relatively more difficult.
# %% Code cell 36
plt.figure(figsize=(10, 6))

# Create a violin plot
sns.violinplot(data=df, x='Customer Type', y='Age', palette=['salmon', 'skyblue'])

# Add labels and title
plt.xlabel('Type of Travel')
plt.ylabel('Age')
plt.title('Age Distribution by Type of Travel')

# Show the plot
plt.show()

# %% [markdown] cell 37
# This graph further elucidates another significant factor contributing to the dissatisfaction of disloyal customers: their age distribution is notably concentrated around the age range of 20 to 40. This aligns closely with the same age range where dissatisfied customers tend to concentrate, as revealed in the previous graph. 
#
# From these observations, it becomes evident that disloyal customers, primarily within the 20 to 40 age bracket, tend to provide lower ratings in inflight entertainment. Consequently, it becomes apparent that focusing on inflight entertainment, particularly for customers within this age group, could be instrumental. By providing an array of compelling and diverse entertainment options, airlines can enhance the satisfaction levels of these customers and attract a broader audience within this age range.
#
# In summary, this insight underscores the importance of catering to the entertainment preferences of customers in the 20 to 40 age group, who are more likely to be disloyal, by offering appealing entertainment choices. This approach has the potential to both increase customer satisfaction and attract more customers within this demographic.
# %% Code cell 38
# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="satisfaction", y="Flight Distance", data=df, palette=['salmon', 'skyblue'])

# Add labels and title
plt.xlabel("Satisfaction")
plt.ylabel("Flight Distance")
plt.title("Flight Distance vs. Satisfaction (Violin Plot)")

# Show the plot
plt.show()

# %% Code cell 39
# Creating a side-by-side histogram subplot for Flight Distance and Satisfaction distribution
plt.figure(figsize=(12, 5))

# Creating the left subplot
plt.subplot(1, 2, 1)
plt.hist(df['Flight Distance'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Flight Distance (miles)')
plt.ylabel('Frequency')
plt.title('Flight Distance Distribution')

# Creating the right subplot
plt.subplot(1, 2, 2)
plt.hist(df['satisfaction'], bins=5, color='salmon', edgecolor='black')
plt.xlabel('Satisfaction')
plt.ylabel('Frequency')
plt.title('Satisfaction Distribution')

# Ensuring proper spacing and layout between subplots
plt.tight_layout()

# Displaying the combined subplots
plt.show()

# %% [markdown] cell 40
# The preceding graphs reveal that the highest flight distances generally fall within the range of approximately 1500 to 2200 miles. 
#
# The dissatisfied customers tend to have a higher density around the mean value, while the density of the remaining distances is relatively low. In contrast, satisfied customers generally have a more evenly distributed density, with a slight increase around the mean.
#
# While there are a few outliers, overall, customer satisfaction does not appear to be significantly influenced by flight distance.
# %% [markdown] cell 41
# # Data preprocessing
# %% [markdown] cell 42
# ### Split the data
# %% Code cell 43
# Split data into x and y.
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

# %% [markdown] cell 44
# ### Encoding and scalling the data
# %% Code cell 45
# One hot Endocing .
X = pd.get_dummies(X, columns=['Class'])

# Label Encoding.
label_encoder = LabelEncoder()
for i in ['Gender', 'Customer Type', 'Type of Travel']:
    X[i] = label_encoder.fit_transform(X[i])

# %% Code cell 46
# Select the features you want to scale
selected_features = X[["Age", "Flight Distance", "Departure Delay in Minutes"]]

# Create a scaler object.
scaler = StandardScaler()

# Fit scaler on the selected features.
scaler.fit(selected_features)

# Transform the selected features with the scaler.
selected_features_scaled = scaler.transform(selected_features)

# Replace the original columns with the scaled values in the DataFrame
X[["Age", "Flight Distance", "Departure Delay in Minutes"]] = selected_features_scaled

X.head()

# %% [markdown] cell 47
# # Modeling
# %% Code cell 48
# Define a list of models to evaluate
models = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# %% Code cell 49

# Split train data into train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over each model and evaluate its accuracy using cross-validation.
for model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Print the mean accuracy score for the current model
    print(f"{model.__class__.__name__}: Mean Accuracy = {scores.mean()}")

# %% [markdown] cell 50
# After evaluating multiple classification models, we found that the Random Forest algorithm demonstrated the best performance based on the mean accuracy metric. Therefore, we selected the Random Forest model to make predictions on our test data.
# %% Code cell 51
# Make tha random forest model with specific best hyperparameters.
model = RandomForestClassifier()

# Fit the model.
model.fit(X_train, y_train)

# Predict y-predict.
y_pred = model.predict(X_test)

# Evaluate the accuracy of y-predict.
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")

# %% [markdown] cell 52
# ### Feature importances
# %% Code cell 53
# Get feature importances
feature_importance = model.feature_importances_

# Sort feature importance indices
sorted_idx = np.argsort(feature_importance)

# Retrieve feature names
features = X.columns

# Plot feature importance
plt.figure()
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color='skyblue')
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance')
plt.show()

# %% [markdown] cell 54
# # Conclusion
# %% [markdown] cell 55
# Our analysis demonstrates that several factors contribute to customer satisfaction in the airline industry. Notably, gender, customer type, and class play a significant role, with women, loyal customers, and those in the business class generally reporting higher satisfaction levels.
#
# Furthermore, rating features such as "Inflight Entertainment" and "Seat Comfort" have emerged as crucial determinants of overall satisfaction. The "Inflight Entertainment" feature, in particular, has shown substantial influence. 
#
# Importantly, we discovered that age also influences satisfaction, with older passengers typically expressing higher satisfaction levels.
#
# In our predictive modeling, the Random Forest algorithm yielded outstanding results, achieving an accuracy of 95%. This algorithm, along with feature importance analysis, confirmed the pivotal role of "Inflight Entertainment" and "Seat Comfort" in shaping customer satisfaction.
#
# In conclusion, this analysis equips Invistico Airlines with insights to prioritize areas for improvement, specifically enhancing Inflight Entertainment and seat comfort. By addressing these aspects, the airline can foster higher customer satisfaction, bolster customer loyalty, and elevate its overall service quality.
