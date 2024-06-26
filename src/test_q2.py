import numpy as np
import matplotlib.pyplot as plt

TOLERANCE = 1e-6 # 収束条件の許容範囲
MAX_ITER = 1000


class OptimizationMethods:
    def __init__(self, A, b):
        self.A = A
        self.b = b
    
    # 勾配を計算する関数
    def gradient(self, w, lambda_):
        return -2 * self.A.T @ (self.b - self.A @ w) + 2 * lambda_ * w
    
    # 目的関数 f(w) を計算する関数
    def f(self, w, lambda_):
        return np.linalg.norm(self.b - self.A @ w)**2 + lambda_ * np.linalg.norm(w)**2
    
    # 最急降下法
    def steepest_descent(self, lambda_, max_iter=MAX_ITER, tol=TOLERANCE):
        _, n = self.A.shape
        w = np.zeros(n)
        L = 2 * (np.linalg.norm(self.A, ord=2)**2 + lambda_)
        step_size = 1 / L
    
        history = []
        for _ in range(max_iter):
            f_val = self.f(w, lambda_)
            history.append(f_val)
    
            grad = self.gradient(w, lambda_)
            w -= step_size * grad
    
            if np.linalg.norm(grad) < tol:
                break
    
        return w, history
    
    # ネステロフの加速勾配法
    def nesterov(self, lambda_, max_iter=MAX_ITER, tol=TOLERANCE):
        _, n = self.A.shape
        w = np.zeros(n)
        y = np.zeros(n)
        L = 2 * (np.linalg.norm(self.A, ord=2)**2 + lambda_)
        step_size = 1 / L
        beta = 0.9  # パラメータの調整
    
        history = []
        for _ in range(max_iter):
            grad = self.gradient(w, lambda_)
            y_next = w - step_size * grad
            w_next = y_next + beta * (y_next - y)
    
            f_val = self.f(w, lambda_)
            history.append(f_val)
    
            if np.linalg.norm(grad) < tol:
                break
    
            y = y_next
            w = w_next
    
        return w, history
    
    # ヘビーボール法
    def heavy_ball(self, lambda_, max_iter=MAX_ITER, tol=TOLERANCE):
        _, n = self.A.shape
        w = np.zeros(n)
        w_prev = np.zeros(n)
        L = 2 * (np.linalg.norm(self.A, ord=2)**2 + lambda_)
        step_size = 1 / L
        beta = 0.9  # パラメータの調整
    
        history = []
        for _ in range(max_iter):
            grad = self.gradient(w, lambda_)
            w_next = w - step_size * grad + beta * (w - w_prev)
    
            f_val = self.f(w, lambda_)
            history.append(f_val)
    
            if np.linalg.norm(grad) < tol:
                break
    
            w_prev = w
            w = w_next
    
        return w, history


# ランダムな A, b を生成し、異なる λ の値をテスト
def main():
    np.random.seed(0)
    m, n = 50, 100
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    lambdas = [0, 1, 10]
    results_steepest_decent = {}
    results_nesterov = {}
    results_heavy_ball = {}
    
    optimization_methods = OptimizationMethods(A, b)
    
    for lambda_ in lambdas:
        # 最急降下法
        _, history_sd = optimization_methods.steepest_descent(lambda_)
        results_steepest_decent[lambda_] = history_sd
        
        # ネステロフの加速勾配法
        _, history_nesterov = optimization_methods.nesterov(lambda_)
        results_nesterov[lambda_] = history_nesterov
        
         # ヘビーボール法
        _, history_heavy_ball = optimization_methods.heavy_ball(lambda_)
        results_heavy_ball[lambda_] = history_heavy_ball
    
    # Plotting the results
    plt.figure(figsize=(12, 8))
    for lambda_ in lambdas:
        plt.plot(results_steepest_decent[lambda_], label=f'Steepest Descent λ={lambda_}')
        plt.plot(results_nesterov[lambda_], label=f'Nesterov λ={lambda_}', linestyle='dashed')
        plt.plot(results_heavy_ball[lambda_], label=f'Heavy-ball λ={lambda_}', linestyle='dotted')
    plt.xlabel('Iteration number k')
    plt.ylabel('f(w_k)')
    plt.yscale("log")
    plt.legend()
    plt.title('Comparison of Steepest Descent, Nesterov, and Heavy-ball Methods')
    plt.savefig("results/test_q2_log.png")
    plt.close()

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for lambda_ in lambdas:
        plt.plot(results_steepest_decent[lambda_], label=f'Steepest Descent λ={lambda_}')
        plt.plot(results_nesterov[lambda_], label=f'Nesterov λ={lambda_}', linestyle='dashed')
        plt.plot(results_heavy_ball[lambda_], label=f'Heavy-ball λ={lambda_}', linestyle='dotted')
    plt.xlabel('Iteration number k')
    plt.ylabel('f(w_k)')
    plt.yscale("linear")
    plt.legend()
    plt.title('Comparison of Steepest Descent, Nesterov, and Heavy-ball Methods')
    plt.savefig("results/test_q2_linear.png")
    plt.close()

if __name__ == "__main__":
    main()
