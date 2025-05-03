import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

# Using pandas to load a smaller dataset
df = pd.read_csv('data/fraud_detection_grafana.csv')  # Make sure the path is correct

#DATA EXPLORING ADN PREPROCESSING
# Checking the first few rows and getting an overview of the dataset
print("📌 Head of the dataset:")
print(df.head(), "\n")

# Checking data types and missing values
print("📌 Info about the dataset:")
print(df.info(), "\n")

print("📌 Missing values in each column:")
print(df.isnull().sum(), "\n")

# Basic statistics about numerical columns
print("📌 Summary statistics for numerical columns:")
print(df.describe(), "\n")


#DATA VISUALIZATION

#Fraud Rate by Transaction Type
fraud_rate_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=fraud_rate_by_type.index, y=fraud_rate_by_type.values, palette="viridis")
plt.title('Fraud Rate by Transaction Type')
plt.ylabel('Fraud Rate (%)')
plt.xlabel('Transaction Type')
plt.show()

#Fraudulent vs Non-Fraudulent Transactions (pie chart)
labels = ['Not Fraud', 'Fraud']
sizes = df['isFraud'].value_counts()
colors = ['#66b3ff','#ff6666']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140)
plt.title('Fraudulent vs Non-Fraudulent Transactions')
plt.show()

#distribution of transaction amounts (histogram)
import matplotlib.ticker as ticker

plt.figure(figsize=(10,5))
sns.histplot(df['amount'], bins=20, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.ticklabel_format(style='plain', axis='x')

plt.show()

#Amount Distribution by Fraud Status
#Kernel Density Estimate (KDE) plot shows the distribution of amounts for both fraudulent and non-fraudulent transactions.
plt.figure(figsize=(12,6))
sns.kdeplot(df[df['isFraud'] == 0]['amount'], shade=True, color='blue', label='Not Fraud')
sns.kdeplot(df[df['isFraud'] == 1]['amount'], shade=True, color='red', label='Fraud')
plt.title('Amount Distribution by Fraud Status')
plt.xlabel('Transaction Amount')
plt.ylabel('Density')
plt.legend()
plt.show()

#Fraud Rate by Time of Transaction (Hourly/Daily)
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
fraud_rate_by_hour = df.groupby('hour')['isFraud'].mean()

plt.figure(figsize=(10,6))
sns.lineplot(x=fraud_rate_by_hour.index, y=fraud_rate_by_hour.values, marker='o', color='purple')
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate')
plt.show()


