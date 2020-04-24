# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载sklearn糖尿病数据集
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 仅使用一维特征
diabetes_X = diabetes_X[:, np.newaxis, 2]

# 将输入分为训练集（前20个数据）和测试集（后20个数据）
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将输出分为训练集（前20个数据）和测试集（后20个数据）
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 创建线性回归对象
regr = linear_model.LinearRegression()

# 使用训练集训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用输入训练集进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)

# 系数
print('Coefficients: \n', regr.coef_)
# 均方误差
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# r2决定系数（模型拟合度越好越接近1，越差越接近0）
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# 输出绘图
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()