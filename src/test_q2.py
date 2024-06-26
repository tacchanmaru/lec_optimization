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

# ネステロフの加速勾配法
def nesterov(A, b, lambda_, max_iter=1000, tol=1e-6):
    m, n = A.shape
    w = np.zeros(n)
    y = np.zeros(n)
    L = 2 * (np.linalg.norm(A, ord=2)**2 + lambda_)
    step_size = 1 / L
    beta = 0.9  # このパラメータは調整可能

    history = []
    for k in range(max_iter):
        grad = gradient(A, b, w, lambda_)
        y_next = w - step_size * grad
        w_next = y_next + beta * (y_next - y)
        
        f_val = f(A, b, w, lambda_)
        history.append(f_val)
        
        if np.linalg.norm(grad) < tol:
            break
        
        y = y_next
        w = w_next
        
    return w, history

# ヘビーボール法
def heavy_ball(A, b, lambda_, max_iter=1000, tol=1e-6):
    m, n = A.shape
    w = np.zeros(n)
    w_prev = np.zeros(n)
    L = 2 * (np.linalg.norm(A, ord=2)**2 + lambda_)
    step_size = 1 / L
    beta = 0.9  # このパラメータは調整可能

    history = []
    for k in range(max_iter):
        grad = gradient(A, b, w, lambda_)
        w_next = w - step_size * grad + beta * (w - w_prev)
        
        f_val = f(A, b, w, lambda_)
        history.append(f_val)
        
        if np.linalg.norm(grad) < tol:
            break
        
        w_prev = w
        w = w_next
        
    return w, history

# ランダムな A, b を生成し、異なる λ の値をテスト
np.random.seed(0)
m, n = 50, 100
A = np.random.randn(m, n)
b = np.random.randn(m)

lambdas = [0, 1, 10]
results = {}
results_nesterov = {}
results_heavy_ball = {}

for lambda_ in lambdas:
    # steepest descent method
    w, history = steepest_descent(A, b, lambda_)
    results[lambda_] = history
    
    # ネステロフの加速勾配法
    _, history_nesterov = nesterov(A, b, lambda_)
    results_nesterov[lambda_] = history_nesterov
    
    # ヘビーボール法
    _, history_heavy_ball = heavy_ball(A, b, lambda_)
    results_heavy_ball[lambda_] = history_heavy_ball


# 結果をプロット
# Plotting the results
plt.figure(figsize=(12, 8))
for lambda_ in lambdas:
    plt.plot(results[lambda_], label=f'Steepest Descent λ={lambda_}')
    plt.plot(results_nesterov[lambda_], label=f'Nesterov λ={lambda_}', linestyle='dashed')
    plt.plot(results_heavy_ball[lambda_], label=f'Heavy-ball λ={lambda_}', linestyle='dotted')
plt.xlabel('Iteration number k')
plt.ylabel('f(w_k)')
plt.yscale('log')
plt.legend()
plt.title('Comparison of Steepest Descent, Nesterov, and Heavy-ball Methods')
plt.savefig("results/test_q2.png")
plt.close()
