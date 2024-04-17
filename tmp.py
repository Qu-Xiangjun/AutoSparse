import math
import random

def rastrigin_function(x):
    """Rastrigin函数"""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])

def simulated_annealing(objective_function, initial_solution, temperature, cooling_rate, stopping_temperature, iterations_per_temperature):
    """
    模拟退火算法

    参数：
    objective_function：目标函数，要求最小值的函数
    initial_solution：初始解
    temperature：初始温度
    cooling_rate：温度衰减率
    stopping_temperature：停止温度
    iterations_per_temperature：每个温度下的迭代次数

    返回值：
    best_solution：找到的最佳解
    """
    current_solution = initial_solution
    best_solution = initial_solution

    while temperature > stopping_temperature:
        for _ in range(iterations_per_temperature):
            # 生成新解
            new_solution = [current_solution[i] + random.uniform(-1, 1) for i in range(len(current_solution))]
            # 计算当前解和新解的成本
            current_cost = objective_function(current_solution)
            new_cost = objective_function(new_solution)
            if new_cost < current_cost:
                # 若新解更优，则接受新解
                current_solution = new_solution
                # 若新解更优，则更新最佳解
                if new_cost < objective_function(best_solution):
                    best_solution = new_solution
            else:
                # 若新解不优，则以一定概率接受新解
                delta = new_cost - current_cost
                probability = math.exp(-delta / temperature)
                if random.random() < probability:
                    current_solution = new_solution
        # 降低温度
        temperature *= cooling_rate

    return best_solution

if __name__ == "__main__":
    dimension = 10  # 定义搜索空间的维度
    # 生成初始解，每个维度的值在[-5.12, 5.12]之间
    initial_solution = [random.uniform(-5.12, 5.12) for _ in range(dimension)]
    temperature = 1000
    cooling_rate = 0.95
    stopping_temperature = 0.1
    iterations_per_temperature = 100

    # 调用模拟退火算法寻找最优解
    best_solution = simulated_annealing(rastrigin_function, initial_solution, temperature, cooling_rate, stopping_temperature, iterations_per_temperature)
    print("最优解:", best_solution)
    print("最优值:", rastrigin_function(best_solution))
