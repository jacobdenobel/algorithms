import numpy as np


def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def compute_hypervolume(X):
    '''This is broken'''
    r = [1.0, 1.0]
    X = np.array(list(sorted(X, key=lambda x:x[0])))
    Q = np.vstack(([float("inf"), r[0]],  X, [r[1], float("inf")],)) 
    hv = np.array(
        [
            (Q[i][0] -  Q[i+1][0] ) *  (Q[i][0] - X[i][0])
         for i in range(len(X))])
    return hv
    

def sms_emoa():
    pop = np.random.random((10, n))
    for j in range(k):
        parent = pop[np.random.randint(0, n)]
        child = parent + sigma * np.random.uniform(0,0.5, n)
        pop_ = np.vstack([pop, child])

        ## random weighing
        w1 = np.random.uniform(0,1)
        w2 = 1 - w1

        f1_ = [w1 * f1(x) for x in pop_] 
        f2_ = [w2 * f2(x) for x in pop_] 

        fitness = np.array(list(zip(f1_, f2_)))        
        mask = is_pareto_efficient(fitness)
        if any(~mask):
            pop = np.delete(pop_, choice(np.where(~mask)[0]), axis=0)
        else: 
            hv = compute_hypervolume(fitness)
            idx = np.argmin(hv)
            # idx = np.random.randint(len(hv))
            pop = np.delete(pop_, idx, axis=0)
            

if __name__ == "__main__":
    np.random.seed(10)
    n       = 10
    k       = 20000
    mu      = 10
    sigma   = .01
    f1      = lambda x: (1 / n**gamma)*(sum(xi**2 for xi in x))**gamma
    f2      = lambda x: (1 / n**gamma)*(sum((1 - xi)**2 for xi in x))**gamma
    for g in (1, .25, .5):
        gamma = g
        sms_emoa()
