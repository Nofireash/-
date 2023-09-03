from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import mglearn

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
"""
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
"""
# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                                 alpha=.8, cmap=mglearn.cm3)
plt.show()


# 做出预测
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_news = np.array([[5, 2.9, 1, 0.2]])
print("X_news.shape: {}".format(X_news.shape))

prediction = knn.predict(X_news)
print("Prediction: {}".format(prediction))
print("Prediction target name: {}".format(iris_dataset['target_names'][prediction]))

# 评估模型
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set predictions:\n {}".format(y_test))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
