#優化最多1
#%%
import random 

population_sizess = 100  #total 100
genome_lengthss = 20    #可以放20格
mutation_ratess = 0.01  #基因突變0.01
crossover_ratess = 0.7 #多少機率交叉 不是僅僅回傳父母
generationss = 200   #重複200次的蝶帶

#隨機定義一個隨機整數 沒有任何突變跟交叉 回傳0跟1 回傳長度為20
def random_genome(length):
    return [random.randint(0 ,1)for _ in range(length)] 

#初始化population 產生基因組
def init_population(population_size,genome_length):
    return [random_genome(genome_length)for _ in range (population_size)]

#
def fitness(genome):
    return sum(genome)
#隨機選擇
def select_parents(population , fitness_values):
    total_fitness = sum(fitness_values)
    pick = random.uniform( 0 , total_fitness)
    current = 0
    for individuals, fitness_value in zip(population , fitness_values):
        current += fitness_value
        if current > pick:
            return individuals
        
def crossover(parent1 , parent2):
    if random.random() < crossover_ratess:
        crossover_point = random.randint( 1,len(parent1) -1)
        return parent1[:crossover_point] + parent2 [crossover_point:],parent2[:crossover_point] + parent1 [crossover_point:]
    else:
        return parent1 , parent2

def mutate(genome):
    for i in range(len(genome)):
        if random.random() < mutation_ratess:
            genome[i] = abs(genome[i]-1)
    return genome


def genetic_algorithm():
    population = init_population(population_sizess , genome_lengthss)

    for generation in range(generationss):
        fitness_values = [fitness(genome) for genome in population]

        new_populations = []
        for _ in range(population_sizess // 2):
            parent1 = select_parents(population , fitness_values)
            parent2 = select_parents(population , fitness_values)
            offspring1 , offspring2 = crossover(parent1,parent2)
            new_populations.extend([mutate(offspring1),mutate(offspring2)])
        population = new_populations
    best_index = fitness_values.index(max(fitness_values))
    best_solutions = population[best_index] 
    print(f"Best solution:{best_solutions}")
    print(f"Best fitness:{fitness(best_solutions)}")



if __name__  == "__main__":
    genetic_algorithm()

    





# %%
import random

population_sizess = 100  # total 100
genome_lengthss = 20     # 可以放20格
mutation_ratess = 0.01   # 基因突變機率 0.01
crossover_ratess = 0.7   # 交叉機率
generationss = 200       # 代數

def random_genome(length):
    return [random.randint(0, 1) for _ in range(length)]

def init_population(population_size, genome_length):
    return [random_genome(genome_length) for _ in range(population_size)]

def fitness(genome):
    return sum(genome)  # 範例適應度：1 的數量越多越好

def select_parents(population, fitness_values):
    total_fitness = sum(fitness_values)
    # 當所有個體適應度為 0 時，隨機選一個避免除以 0 或 pick 超出範圍
    if total_fitness == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness_value in zip(population, fitness_values):
        current += fitness_value
        if current >= pick:
            return individual
    # fallback（理論上不會到這）
    return population[-1]

def crossover(parent1, parent2):
    if random.random() < crossover_ratess:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        # 回傳複本以避免後續 mutate 直接改到 parent
        return parent1.copy(), parent2.copy()

def mutate(genome):
    genome = genome.copy()  # copy 避免就地修改外部變數
    for i in range(len(genome)):
        # 呼叫 random.random()（記得加括號）
        if random.random() < mutation_ratess:
            genome[i] = 1 - genome[i]  # flip bit
    return genome

def genetic_algorithm():
    # 正確的變數名稱 population（原本拼錯為 popualtion）
    population = init_population(population_sizess, genome_lengthss)

    for generation in range(generationss):
        fitness_values = [fitness(genome) for genome in population]

        new_populations = []
        for _ in range(population_sizess // 2):
            parent1 = select_parents(population, fitness_values)
            parent2 = select_parents(population, fitness_values)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_populations.extend([mutate(offspring1), mutate(offspring2)])

        population = new_populations

    # 最後一代計算 fitness（修正：必須用最後 population 的 fitness）
    fitness_values = [fitness(genome) for genome in population]
    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {fitness(best_solution)}")

if __name__ == "__main__":
    genetic_algorithm()
# %%
import numpy as np

# 目標函數（越小越好）
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# GA 參數
POP_SIZE = 50
GENS = 200
DIM = 2  # 基因維度
BOUND = [-5.12, 5.12]
CX_RATE = 0.8
MUT_RATE = 0.1
ELITE_NUM = 2

# 初始化族群
pop = np.random.uniform(BOUND[0], BOUND[1], (POP_SIZE, DIM))

# 計算適應度（因為我們是最小化，所以取負值）
def fitness(ind):
    return -rastrigin(ind)

# 錦標賽選擇
def select(pop, k=3):
    candidates = np.random.choice(len(pop), k)
    best = max(candidates, key=lambda idx: fitness(pop[idx]))
    return pop[best]

# 單點交叉
def crossover(p1, p2):
    if np.random.rand() < CX_RATE:
        point = np.random.randint(1, DIM)
        child1 = np.concatenate((p1[:point], p2[point:]))
        child2 = np.concatenate((p2[:point], p1[point:]))
        return child1, child2
    return p1.copy(), p2.copy()

# 高斯突變
def mutate(ind):
    for i in range(DIM):
        if np.random.rand() < MUT_RATE:
            ind[i] += np.random.normal(0, 0.5)
            ind[i] = np.clip(ind[i], BOUND[0], BOUND[1])
    return ind

# 主迴圈
for gen in range(GENS):
    # 評估 & 排序
    pop = sorted(pop, key=lambda ind: fitness(ind), reverse=True)
    new_pop = pop[:ELITE_NUM]  # 精英保留

    # 生成新族群
    while len(new_pop) < POP_SIZE:
        p1, p2 = select(pop), select(pop)
        c1, c2 = crossover(p1, p2)
        new_pop.append(mutate(c1))
        if len(new_pop) < POP_SIZE:
            new_pop.append(mutate(c2))

    pop = np.array(new_pop)

    # 每代顯示最佳解
    best = pop[0]
    print(f"Gen {gen+1}: best = {best}, f = {rastrigin(best):.4f}")

print("\n最終最佳解：", pop[0], "函數值：", rastrigin(pop[0]))
# %%
