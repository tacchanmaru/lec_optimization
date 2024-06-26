import numpy as np
import matplotlib.pyplot as plt

# 勾配を計算する関数
def gradient(A, b, w, lambda_):
    return -2 * A.T @ (b - A @ w) + 2 * lambda_ * w

# 目的関数 f(w) を計算する関数
def f(A, b, w, lambda_):
    return np.linalg.norm(b - A @ w)**2 + lambda_ * np.linalg.norm(w)**2

# 最急降下法
def steepest_descent(A, b, lambda_, max_iter=1000, tol=1e-6):
    m, n = A.shape
    w = np.zeros(n)
    L = 2 * (np.linalg.norm(A, ord=2)**2 + lambda_)
    step_size = 1 / L

    history = []
    for k in range(max_iter):
        f_val = f(A, b, w, lambda_)
        history.append(f_val)
        
        grad = gradient(A, b, w, lambda_)
        w -= step_size * grad
        
        if np.linalg.norm(grad) < tol:
            break
            
    return w, history

# ランダムな A, b を生成し、異なる λ の値をテスト
np.random.seed(0)
m, n = 50, 100
A = np.random.randn(m, n)
b = np.random.randn(m)

lambdas = [0, 1, 10]
results = {}

for lambda_ in lambdas:
    w, history = steepest_descent(A, b, lambda_)
    results[lambda_] = history

# 結果をプロット
plt.figure(figsize=(12, 8))
for lambda_, history in results.items():
    plt.plot(history, label=f'λ={lambda_}')
plt.xlabel('Iteration number k')
plt.ylabel('f(w_k)')
plt.yscale('log')
plt.legend()
plt.title('Steepest Descent Method for different λ')
plt.savefig("results/test_q1.png")
plt.close()