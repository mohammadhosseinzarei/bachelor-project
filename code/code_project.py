import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000, penalty=None):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.penalty = penalty

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        tol = 1e-5
        prev_loss = 0
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)))
            # add regularization derevitive terme
            if self.penalty == 'l1':
                dw += self.lr * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += self.lr * 2 * self.weights
            db = (1 / n_samples) * (np.sum(y_pred - y))
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

            current_loss = np.mean(np.square(y_pred - y))

            if abs(current_loss - prev_loss) < tol:
                break

            prev_loss = current_loss

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

import numpy as np
np.random.seed(0)  # برای تکرارپذیری
X = 2 * np.random.rand(100, 1)  # 100 نمونه با یک ویژگی
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + نویز

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# آموزش مدل
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
# پیش‌بینی
plt.figure(figsize=(10, 6))
# plt.subplot(1,1,1)
plt.scatter(X, y, color='blue', label="data")
plt.plot(X, y_pred, color='red', label="y_pred")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
#############################Logistic Regression###################################
class Reg_logstic():
    def __init__(self, X, y, num_iter=1000, alpha=0.01):
        self.X = X
        self.y = y
        self.num_iter = num_iter
        self.alpha = alpha
        self.intercept = np.ones((X.shape[0], 1))

    def add_intercept(self):
        return np.concatenate((self.intercept, self.X), axis=1)

    def calc_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calc_h(self, X, theta):
        z = np.dot(X, theta)
        return self.calc_sigmoid(z)

    def cost_function(self):
        XX = self.add_intercept()
        theta = np.zeros(XX.shape[1])
        m = self.y.size
        cost_list = []

        for i in range(self.num_iter):
            h = self.calc_h(XX, theta)
            cost = (-self.y * np.log(h) - (1 - self.y) * np.log(1 - h)).mean()
            cost_list.append(cost)

            gradient = np.dot(XX.T, (h - self.y)) / m
            theta -= self.alpha * gradient

            if i % 100 == 0:
                print(f"Iteration {i} | Cost: {cost}")

        print(f"Final Cost: {cost}")
        print(f"Coefficients: {theta}")
        return theta, cost_list
    
    
X, y = make_classification(
        n_samples=100, 
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1, 
        n_clusters_per_class=1)


Rel=Reg_logstic(X, y, num_iter=1000, alpha=0.01)
XX=Rel.add_intercept()
theta = np.zeros(XX.shape[1])
h=Rel.calc_h(XX,theta)
cost=Rel.cost_function()
print(XX)

# k-means
import numpy as np
import matplotlib.pyplot as plt

# --- تابع میانگین مرکز خوشه (از خودت) ---
def centroid(x):
    length_x = x.shape[0]
    sum_x = np.sum(x, axis=0)
    return sum_x / length_x

# --- تابع فاصله اقلیدسی از صفر ---
def euclidean_distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** 0.5

# --- تابع پیدا کردن نزدیک‌ترین مرکز خوشه (کاملاً فرام اسکرچ) ---
def closest_centroid(x, centroids):
    min_dist = float('inf')
    closest_index = -1

    for i in range(len(centroids)):
        dist = euclidean_distance(x, centroids[i])
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    return closest_index

# --- کلاس KMeans از صفر ---
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # انتخاب تصادفی مراکز اولیه
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # اختصاص هر نقطه به نزدیک‌ترین مرکز
            labels = []
            for point in X:
                c = closest_centroid(point, self.centroids)
                labels.append(c)
            labels = np.array(labels)

            # به‌روزرسانی مراکز خوشه با تابع centroid()
            new_centroids = []
            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(centroid(cluster_points))
                else:
                    new_centroids.append(self.centroids[i])  # اگر خوشه خالی بود، همان مرکز قبلی حفظ شود
            new_centroids = np.array(new_centroids)

            # بررسی همگرایی
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        preds = []
        for point in X:
            preds.append(closest_centroid(point, self.centroids))
        return np.array(preds)

# --- تست الگوریتم ---
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans = KMeans(k=3)
kmeans.fit(X)

# --- نمایش نتایج ---
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200)
plt.title("K-Means Clustering (Fully From Scratch)")
plt.show()



