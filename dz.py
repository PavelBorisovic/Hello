"""Задание №17-18"""
"График №1"
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('flight_delays.csv', sep=',')
for i in range(1,13,1):
    data.loc[data["Month"] == 'c-'+str(i), "Month"] = i
data.sort_values(by='Month')
x1 = data[data['dep_delayed_15min'] == 'Y'].groupby('Month').Month.count()
x2 = data.groupby('Month').Month.count()
x3 = x1 / x2
x3.plot(kind='bar', x='Month', y='count')
plt.show()

"График №2"
warnings.filterwarnings('ignore')

data = pd.read_csv('flight_delays.csv', sep=',')
x1 = data[data['dep_delayed_15min'] == 'Y'].groupby('Distance').Distance.count()
x1.plot(kind='line', x='Distance', y='count')
plt.show()

"График №3"
warnings.filterwarnings('ignore')

data = pd.read_csv('flight_delays.csv', sep=',')
x1 = data[data['dep_delayed_15min'] == 'Y'].groupby('Dest').dep_delayed_15min.count()
D = x1.nlargest(5)
print(D)
D.plot(kind='pie')
plt.show()

"График №4"
warnings.filterwarnings('ignore')

data = pd.read_csv('flight_delays.csv', sep=',')
for i in range(1, 13, 1):
    if i in [1, 2, 12]:
        data.loc[data["Month"] == 'c-'+str(i), "Month"] = 'Winter'
    elif i in [3, 4, 5]:
        data.loc[data["Month"] == 'c-' + str(i), "Month"] = 'Spring'
    elif i in [6, 7, 8]:
        data.loc[data["Month"] == 'c-'+str(i), "Month"] = 'Summer'
    else:
        data.loc[data["Month"] == 'c-'+str(i), "Month"] = 'Autumn'
data = data.rename(columns={"Month": "Seasons"})
x1 = data[data['dep_delayed_15min'] == 'Y'].groupby('Seasons').dep_delayed_15min.count()
print(x1)
x1.plot(kind='pie', colors = ("orangered", "limegreen", "magenta", "royalblue"))
plt.show()

"Графики №5-6"
warnings.filterwarnings('ignore')

data = pd.read_csv('flight_delays.csv', sep=',')
x1 = data[data['dep_delayed_15min'] == 'Y'].groupby('UniqueCarrier').dep_delayed_15min.count()
B = x1.nlargest(10)
N = x1.nsmallest(10)
print(B)
B.plot(x="UniqueCarrier", kind="bar", color = 'darkorange')
plt.show()
N.plot(x="UniqueCarrier", kind="bar", color = 'olive')
plt.show()



"""Задание №21"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.20, random_state=5)

neigh = KNeighborsClassifier(n_neighbors=18)
neigh.fit(X_train, y_train)
print(classification_report(y_test, neigh.predict(X_test)))


scores = cross_val_score(neigh, iris_dataset['data'], iris_dataset['target'], cv=10, scoring='accuracy')
print(scores)
print(scores.mean())
