import numpy as np

class TSP:
    def __init__(self, n_cities):
        self.n_cities = n_cities
        self.cities = np.arange(self.n_cities)
        self.yopt = n_cities - 1
        self.n_evals = 0

    def __call__(self, order):
        self.n_evals += 1
        return np.abs(np.diff(order)).sum()


class TSPSolverMixin:
    def init(self, problem: TSP, n):
        self.population = np.vstack([
            np.random.permutation(problem.cities)
            for _ in range(n)
        ]) 
        self.f = np.array([problem(x) for x in self.population])
        self.yopt, self.xopt = float("inf"), None

    def swap(self, x, n_swaps):
        n_swaps = min(int(len(x) // 2), n_swaps)
        swaps = np.random.choice(len(x), n_swaps*2, replace=False)
        source = swaps[:n_swaps]
        target = swaps[n_swaps:]
        x[target], x[source] = x[source], x[target]
        return x

    def select(self, n):
        idx = np.argsort(self.f)[:n]
        return self.population[idx, :], self.f[idx]

    def break_conditions(self, g, problem: TSP):
        if min(self.f) < self.yopt:
            self.yopt = min(self.f)
            self.xopt = self.population[np.argmin(self.f)] 

        return self.yopt == problem.yopt
        

class GeneticAlgorithm(TSPSolverMixin):
    def __init__(self, mu, lambda_, max_iter, crossover_method=1):
        self.mu = mu
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.crossover_method = self.no_crossover \
            if crossover_method is None else self.pmw_crossover

    def __call__(self, problem: TSP):
        pm = 3 / problem.n_cities
        self.init(problem, self.mu)
        
        for g in range(self.max_iter):     
            new_population = self.crossover_method()
            n_swaps = (
                pm > np.random.uniform(size=(problem.n_cities, self.lambda_))
            ).sum(axis=0)

            for xi in range(len(new_population)):
                new_population[xi] = self.swap(new_population[xi], n_swaps[xi])
                self.f = np.append(self.f, problem(new_population[xi]))
                
            self.population = np.vstack([self.population, new_population])
            self.population, self.f = self.select(self.mu)

            if self.break_conditions(g, problem):
                break

        return self.yopt, self.xopt

    def no_crossover(self):
        return self.population[
            np.random.choice(len(self.population), self.lambda_), :
        ]

    def pmw_crossover(self):
        new_population = []
        for _ in range(self.lambda_ // 2):
            x1, x2 = np.random.choice(len(self.population), 2)
            p1, p2 = self.population[x1].copy(), self.population[x2].copy()
            c1, *_ = np.random.choice(len(self.population), 1)
            for i, (pi1, pi2) in enumerate(zip(p1, p2)):
                if i > c1:
                    break
                p1l, *_ = np.where(self.population[x1] == pi2)
                p2l, *_ = np.where(self.population[x2] == pi1)
                p1[p1l], p1[i] = p1[i], p1[p1l]
                p2[p2l], p2[i] = p2[i], p2[p2l]
            new_population.append(p1)
            new_population.append(p2)

        return np.vstack(new_population[:self.lambda_])


class PlantPropagationAlgorithm(TSPSolverMixin):

    def __init__(self, n, max_offspring, max_iter):
        self.n = n
        self.max_iter = max_iter
        self.max_offspring = max_offspring
        self.smax = 10

    def __call__(self, problem: TSP):
        self.init(problem, self.n)

        for g in range(self.max_iter):     
            fmax, fmin = np.max(self.f), np.min(self.f)
            z = (fmax - self.f) / (fmax - fmin)\
                 if not fmax == fmin else np.ones(len(self.f)) * 5
            F = .5 * (np.tanh(4 * z - 2) + 1)
            n = np.ceil(
                self.max_offspring * F * np.random.uniform(size=self.n)
            ).astype(int)
            
            new_population = []
            for ni, F, x in zip(n, F, self.population):
                for si in range(ni):
                    n_swaps = np.ceil(self.smax * ((1 - F) * np.random.uniform())).astype(int)
                    offspring = self.swap(x.copy(), n_swaps)
                    new_population.append(offspring)
                    self.f = np.append(self.f, problem(offspring))
            
            self.population = np.vstack([self.population, new_population])
            self.population, self.f = self.select(self.n)

            if self.break_conditions(g, problem):
                break
          
        return self.yopt, self.xopt
       

def run(alg, n_reps=25, seed=10, dim=10):
    evals, n_succ = [], 0
    for rep in range(n_reps):
        if seed:
            np.random.seed(seed+rep)
        tsp = TSP(dim)    
        y, _ = alg(tsp)
        n_succ += int(y == tsp.yopt)
        evals.append(tsp.n_evals)

    return (np.sum(evals) / n_succ) if n_succ else float("inf"), np.sum(evals) / n_reps   


if __name__ == '__main__':
    print("ga", run(GeneticAlgorithm(5, 10, 1_000)))
    print("ppa", run(PlantPropagationAlgorithm(5, 6, 1_000)))
