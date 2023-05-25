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

"""Задание 16"""
"Задача 1"
import numpy as np
"1"
l = np.arange(12, 43)

"2"
h = np.zeros(10)
h[4] = 1

"3"
A = np.arange(0, 9)
B = A.reshape(3,3)

"4"
h = np.array([1,2,0,0,4,0])
print("Индексы положительных элементов: \n", np.where(h > 0))

"5"
f = np.random.random((5,3))
v = np.random.random((3,2))
print (f.dot(v))

"6"
A = np.zeros((10,10))
A[1:-1,1:-1] = 1

"7"
l = np.random.random(5)
l.sort()

"8"
a = np.random.random(5)
for ind, val in np.ndenumerate(a):
  print(ind, val)

"Задача 2"
import numpy as np
import imageio
import matplotlib.pyplot as plt

img = imageio.imread()
h = np.array([0.299, 0.587, 0.114, 1])
q = img.dot(h)
plt.imshow(q)
plt.show()

"Задача 3"
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import math
"1"
b1 = math.sin(1/5)*math.exp(1/10)+5*math.exp(-1/2)
b2 = math.sin(15/5)*math.exp(15/10)+5*math.exp(-15/2)
b = np.array([[b1],[b2]])
A = np.array([[1,1],[1,15]])
sol = linalg.solve(A,b)
print(sol)
f = lambda x: sol[0]+sol[1]*x
plt.plot(x, f(x))

"2"
b3 = math.sin(8/5)*math.exp(8/10)+5*math.exp(-8/2)
b = np.array([[b1],[b2],[b3]])
A = np.array([[1,1,1],[1,15,225],[1,8,64]])
sol = linalg.solve(A,b)
print(sol)
d = lambda x: sol[0]+sol[1]*x+sol[2]*x**2
plt.plot(x, d(x))

"3"
b3 = math.sin(4/5)*math.exp(4/10)+5*math.exp(-4/2)
b4 = math.sin(10/5)*math.exp(10/10)+5*math.exp(-10/2)
b = np.array([[b1],[b2],[b3],[b4]])
A = np.array([[1,1,1,1],[1,15,225,15**3],[1,4,16,64],[1,10,100,1000]])
sol = linalg.solve(A,b)
print(sol)
t = lambda x: sol[0]+sol[1]*x+sol[2]*x*x+sol[3]*x*x*x
plt.plot(x, t(x))
plt.show()
