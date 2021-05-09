from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X,y = load_boston(return_X_y=True)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsRegressor(n_neighbors=6)

knn.fit(X, y)
score = knn.score(X, y)
print(score)