import pandas as pd

df = pd.read_csv("data\\nfhs_data.csv")

print(df.head())
print(df.shape)
df.info()
df.isnull().sum()
df.columns
df.describe()

df.columns = df.columns.str.strip().str.replace(" ", "_")
print(df.columns.tolist())
district_col = "District_Names"
population_col = "Number_of_Households_surveyed"
alcohol_col = "Men_age_15_years_and_above_who_consume_alcohol_(%)"

print(df.columns)

df.isnull().sum().sort_values(ascending=False)
df = df.fillna(0)

for col in df.columns:
    print(col)

df.sort_values(by=population_col, ascending=False).head(10)
df.sort_values(by=alcohol_col, ascending=False).head(10)

state_group = df.groupby(df.columns[0]).mean(numeric_only=True)
print(state_group.head())

import matplotlib.pyplot as plt

top10 = df.sort_values(by=population_col, ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.bar(top10[district_col], top10[population_col])
plt.xticks(rotation=90, ha = 'right')
plt.title("Top 10 Districts by Population")
plt.xlabel("District")
plt.ylabel("Households")
plt.show()

plt.figure()
df[alcohol_col].hist()
plt.title("Alcohol Consumption Distribution")
plt.tight_layout()
plt.show()

for i, col in enumerate(df.columns):
    print(i, col)

df.sort_values(by="Men_age_15_years_and_above_who_consume_alcohol_(%)", ascending=False).head(10)
df.sort_values(by="Men_age_15_years_and_above_who_consume_alcohol_(%)", ascending=True).head(10)

state_analysis = df.groupby("State/UT").mean(numeric_only=True)

state_analysis.sort_values(
    by="Men_age_15_years_and_above_who_consume_alcohol_(%)",
    ascending=False
).head(10)

corr = df.corr(numeric_only=True)

print(corr["Men_age_15_years_and_above_who_consume_alcohol_(%)"].sort_values(ascending=False))

plt.figure()
plt.scatter(
    df["Number_of_Households_surveyed"],
    df["Men_age_15_years_and_above_who_consume_alcohol_(%)"]
)
plt.xlabel("Households")
plt.ylabel("Alcohol %")
plt.title("Alcohol vs Population")
plt.show()

import seaborn as sns

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="RdBu_r",   
    center=0,        
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()

import numpy as np
from scipy import stats

data = df["Men_age_15_years_and_above_who_consume_alcohol_(%)"]

sample_mean = np.mean(data)
population_mean = 30
std_dev = np.std(data)
n = len(data)

z = (sample_mean - population_mean) / (std_dev / np.sqrt(n))

p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print("Z-score:", z)
print("P-value:", p_value)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)

plt.figure(figsize=(8,5))
plt.plot(x, y, label="Normal Distribution")

plt.axvline(z, color='red', linestyle='--', label=f'Z = {round(z,2)}')

plt.title("Z-Test Visualization")
plt.legend()
plt.show()

state1 = df[df["State/UT"] == df["State/UT"].unique()[0]][alcohol_col]
state2 = df[df["State/UT"] == df["State/UT"].unique()[1]][alcohol_col]

t_stat, p_val = stats.ttest_ind(state1, state2)

print("T-statistic:", t_stat)
print("P-value:", p_val)

plt.figure(figsize=(8,5))

plt.boxplot([state1, state2], labels=["State 1", "State 2"])

plt.title("T-Test: State Comparison")
plt.ylabel("Alcohol %")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Target
y = df["Men_age_15_years_and_above_who_consume_alcohol_(%)"]

# Features (numeric columns only)
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[y.name])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Linear Regression)")

plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         color='red')  

plt.tight_layout()
plt.show()






