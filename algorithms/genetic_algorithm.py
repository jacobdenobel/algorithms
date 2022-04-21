import ioh
import numpy as np

class GeneticAlgorithm:
    def __init__(
        self, max_iterations=10_000, mu=5, lambda_=21, mu_plus_lambda=True, pm=None
    ):
         
        self.max_iterations: int = max_iterations
        self.mu = mu
        self.lambda_ = lambda_
        self.mu_plus_lambda = mu_plus_lambda
        self.pm = pm

        if mu < 1 or lambda_ < 1:
            raise ValueError("Both mu and lambda should be larger than 1")

        if not mu_plus_lambda and not (lambda_ >= mu):
            raise ValueError("lambda should be >= mu when not using mu_plus_lambda")

    def __call__(self, problem: ioh.problem.Integer) -> None:
        dim = problem.meta_data.n_variables
        pm = self.pm or (1 / dim)

        # Initialize population
        parents = np.random.choice([0, 1], size=(self.mu, dim))
        fitness = np.array([problem(x) for x in parents])

        for g in range(self.max_iterations):
            # Select mu parents (Rank selection)
            idx = np.argsort(fitness)[::-1][: self.mu]
            fitness = fitness[idx]
            parents = parents[idx, :]

            # Recombine lambda offspring (1-point crossover)
            pidx = np.random.choice(range(self.mu), size=self.lambda_ * 2)
            cidx = np.random.choice(range(1, dim - 1), size=self.lambda_)
            offspring = np.array(
                [
                    np.r_[parents[p1, :c], parents[p2, c:]].copy()
                    for c, p1, p2 in zip(cidx, pidx[::2], pidx[1::2])
                ]
            )

            # Mutate offspring (bit-flip mutation)
            mask = np.random.choice([False, True], size=offspring.shape, p=[1 - pm, pm])
            offspring[mask] = np.abs((offspring[mask] - 1))

            # Compute fitness
            offspring_fitness = np.array([problem(x) for x in offspring])

            if self.mu_plus_lambda:
                parents = np.vstack([parents, offspring])
                fitness = np.r_[fitness, offspring_fitness]
            else:
                parents = offspring
                fitness = offspring_fitness

            if problem.state.optimum_found:
                break


def decode(x):
    s = "".join(map(str, x[1:]))
    m = int(len(x) / 2)
    return dict(mu_plus_lambda=bool(x[0]), mu=int(s[:m], 2), lambda_=int(s[m:], 2))


cache = {}
def cache_function(f):
    def inner(x):
        key = decode(x).values()
        if not cache.get(key):
            cache[key] = f(x)
        print(key, cache[key])
        return cache[key]
    return inner

@cache_function
def objective(x):
    np.random.seed(32)
    d = decode(x)
    try:
        algorithm = GeneticAlgorithm(**d)
    except:
        return -np.inf

    problem = ioh.get_problem(2, 1, 100, "Integer")
    algorithm(problem)

    return -problem.state.evaluations


def meta_ga():
    problem = ioh.problem.wrap_integer_problem(
        objective,
        "solve_ga",
        13,
        ioh.OptimizationType.Maximization,
        ioh.IntegerConstraint([0], [1]),
    )
    algorithm = GeneticAlgorithm(250)
    algorithm(problem)

    print(-problem.state.current_best.y)
    print(decode(problem.state.current_best.x))

def main():
    # Set a random seed in order to get reproducible results
    seed = 32
    np.random.seed(seed)

    # Get a problem from the IOHexperimenter environment
    n_variables = 100
    problem: ioh.problem.Integer = ioh.get_problem(2, 1, n_variables, "Integer")

    # Run the algoritm on the problem
    algorithm = GeneticAlgorithm()
    algorithm(problem)

    # Inspect the results
    print("Best solution found:")
    print("".join(map(str, problem.state.current_best.x)))
    print("With an objective value of:", problem.state.current_best.y)
    print(f"Used {problem.state.evaluations} evaluations")
    print()


if __name__ == "__main__":
    main()
