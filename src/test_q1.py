import numpy as np
import matplotlib.pyplot as plt

TOLERANCE = 1e-6 # 収束条件の許容範囲


class SteepestDescent:
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
    def steepest_descent(self, lambda_, max_iter=1000, tol=TOLERANCE):
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


# ランダムな A, b を生成し、異なる λ の値をテスト
def main():
    np.random.seed(0)
    m, n = 50, 100
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    lambdas = [0, 1, 10]
    results = {}

    steepest_descent_solver = SteepestDescent(A, b)

    for lambda_ in lambdas:
        w, history = steepest_descent_solver.steepest_descent(lambda_)
        results[lambda_] = history

    # 結果をプロット
    plt.figure(figsize=(12, 8))
    for lambda_, history in results.items():
        plt.plot(history, label=f'λ={lambda_}')
    plt.xlabel('Iteration number k')
    plt.ylabel('f(w_k)')
    plt.yscale('linear')  # y軸のスケールをlinearに設定
    plt.legend()
    plt.title('Steepest Descent Method for different λ')
    plt.savefig("results/test_q1_linear.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    for lambda_, history in results.items():
        plt.plot(history, label=f'λ={lambda_}')
    plt.xlabel('Iteration number k')
    plt.ylabel('f(w_k)')
    plt.yscale('log')  # y軸のスケールをlinearに設定
    plt.legend()
    plt.title('Steepest Descent Method for different λ')
    plt.savefig("results/test_q1_log.png")
    plt.close()

if __name__ == "__main__":
    main()