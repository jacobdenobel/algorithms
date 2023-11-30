# Jacob's Algorithm Collection

This repository houses a collection of various black box optimization algorithms implemented in Python. These algorithms are designed to optimize objective functions with limited knowledge about the function itself. The project builds upon IOHexperimenter, a toolkit for black box optimization experimentation. Please note that the algorithms included in this repository are implemented without any tests or guarantees, and based on descriptions from papers, but might deviate, and their usage comes with certain risks. Support for this project is limited.

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithms](#algorithms)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction

Black box optimization refers to a class of optimization problems where the objective function is not explicitly known, and only its evaluations are available. These algorithms are useful in scenarios where the underlying objective function is costly to evaluate or difficult to model. This repository provides a collection of black box optimization algorithms that you can experiment with.

## Algorithms

The repository currently includes the following black box optimization algorithms:

- **Genetic Algorithm**
- **Evolution Strategy (1+1) 1/5 success**
- **Evolution Strategy (self-adaptive)**
- **Evolution Strategy (Schwefel)**
- **CSA-ES (Evolution Strategy with Cumulative Stepsize Adaptation)**
- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
- **MA-ES (Matrix Adaptation Evolution Strategy)**
- **DR1 (Derandomized ES)**
- **DR2 (Derandomized ES)**
- **DR3 (Derandomized (GSA) ES)**
- **Evolutionary Gradient Search**
- **Unbounded Integer EA**
- **Differential Evolution**
- **Global Simple Evolutionary Multi-objective Algorithm (GSEMO)**
- **Particle Swarm Optimization (PSO)**
- **Plant Propagation Algorithm**
- **Augmented Random Search**
- **(Self)-Guided ES** (Based on: https://github.com/IJCAI2020-SGES/SGES)

Feel free to explore and experiment with these algorithms.

## Getting Started

Before using these algorithms, you should have Python and the required dependencies installed. We recommend using a virtual environment to manage your dependencies. To get started, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/jacobdenobel/algorithms.git
   ```

2. Navigate to the project directory:

   ```bash
   cd algorithms
   ```

3. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The implementation of these algorithms is ad hoc, and you can experiment with them by importing and applying them to your specific optimization problems, some tweaks might be needed to make them work for your specific use case. I recommend exploring and reading the source code for implementation details.

Please keep in mind the following:

- The algorithms are not guaranteed to work for every optimization problem.
- Ensure you have an understanding of your specific problem and the algorithm you intend to use.
- Use these algorithms at your own risk, as they might not be suitable for all use cases.

## Contributing

Contributions to this project are welcome. If you'd like to improve existing algorithms, add new ones, or fix issues, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit them.
4. Push your changes to your fork: `git push origin feature/your-feature-name`.
5. Create a pull request on this repository.

Your contributions will be reviewed and merged if they align with the project's goals.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Disclaimer**: Please note that the algorithms in this repository are provided as-is, and the maintainers do not provide extensive support for them. Use them responsibly and at your own discretion.

---

Feel free to customize the README further to provide more details or information specific to your project. Good luck with your black box optimization algorithms repository!
