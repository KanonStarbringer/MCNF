# Multi-Commodity Network Flow Problem (MCNFP) - Column Generation with Bellman-Ford

This project implements a solution for the **Multi-Commodity Network Flow Problem (MCNFP)** using the **Column Generation** algorithm with support for the **Bellman-Ford** algorithm to handle negative reduced costs.

## 📋 Description

The MCNFP is a classic network optimization problem where multiple commodities need to be transported through a network, respecting arc capacities and minimizing total cost. This code implements:

- **Random instance generation** for the MCNFP problem
- **Feasibility checking** using Linear Programming
- **Column Generation solution** with Bellman-Ford algorithm
- **Detailed execution reports** generation

### Key Features

- ✅ Support for multiple sources and sinks per commodity
- ✅ Use of Bellman-Ford when there are negative reduced costs (necessary due to dual multipliers)
- ✅ Automatic initial path generation
- ✅ Infeasibility handling with slack variables
- ✅ Complete summary file generation (`CGA_Summary.txt`)

## 🔧 Requirements

### Dependencies

The code requires the following Julia packages:

```julia
using Random, Graphs, Printf, JuMP, Gurobi, MathOptInterface
```

### Installing Dependencies

To install the required dependencies, run in the Julia REPL:

```julia
import Pkg
Pkg.add(["Graphs", "JuMP", "Gurobi", "MathOptInterface"])
```

**Note:** The Gurobi solver requires a valid license. Alternatively, you can modify the code to use another JuMP-compatible solver (such as `HiGHS`, `Cbc`, etc.).

## 🚀 How to Use

### 1. Generate an Instance

```julia
instance = generate_mcnfp_instance(
    num_nodes=10,        # Number of nodes in the network
    num_commodities=3,    # Number of commodities
    density=0.4,          # Graph density (0.0 to 1.0)
    seed=123,            # Seed for reproducibility
    max_cap=100.0,       # Maximum arc capacity
    max_cost=20.0,       # Maximum cost per arc
    max_demand=10.0      # Maximum demand per commodity
)

print_instance_summary(instance)
```

### 2. Check Feasibility

Before solving, it is recommended to check if the instance is feasible:

```julia
is_possible = check_feasibility_lp(instance; verbose=true)
```

### 3. Solve with Column Generation

```julia
result = solve_mcnfp_column_generation(
    instance;
    max_iterations=1000,  # Maximum number of iterations
    verbose=true,          # Display detailed log
    tolerance=1e-6        # Tolerance for reduced costs
)

# Check result
if result.optimal
    println("✅ OPTIMAL solution found!")
    println("Objective value: ", result.objective_value)
    println("Columns generated: ", result.columns_generated)
else
    println("⚠️  Non-optimal solution")
    println("Objective value: ", result.objective_value)
end
```

## 📁 Code Structure

### Data Structures

- **`Commodity`**: Represents a commodity with its sources, sinks, and demand
- **`MCNFPInstance`**: Contains the graph, capacities, costs, and commodities

### Main Functions

- **`generate_mcnfp_instance()`**: Generates random instances of the problem
- **`check_feasibility_lp()`**: Checks feasibility using Linear Programming
- **`solve_mcnfp_column_generation()`**: Solves the problem using column generation
- **`dijkstra_shortest_path()`**: Finds shortest path (for non-negative costs)
- **`bellman_ford_shortest_path()`**: Finds shortest path (supports negative costs)
- **`generate_summary_file()`**: Generates execution summary file

## 🔍 Column Generation Algorithm

The implemented algorithm follows these steps:

1. **Initialization**: Finds initial paths for each commodity using Dijkstra or BFS
2. **Restricted Master Problem (RMP)**: Solves the optimization problem with current paths
3. **Pricing Subproblem**: For each commodity, finds new paths with negative reduced cost
   - Uses **Dijkstra** when all reduced costs are non-negative
   - Uses **Bellman-Ford** when there are negative reduced costs
4. **Column Addition**: Adds promising new paths to the model
5. **Convergence**: Stops when there are no more paths with negative reduced cost

### Why Bellman-Ford?

During pricing, reduced costs can be negative even when original costs are positive, due to dual multipliers from capacity constraints. The Dijkstra algorithm does not work correctly with negative costs, so the code automatically detects when to use Bellman-Ford.

## 📊 Output

The algorithm generates a `CGA_Summary.txt` file containing:

- Instance summary (nodes, arcs, commodities)
- Complete execution log
- Final result (optimal or not)
- Objective value
- Number of iterations and columns generated
- Detailed solution (flow per path and per arc)

## ⚙️ Configurable Parameters

### Instance Generation

- `num_nodes`: Number of nodes (default: 10)
- `num_commodities`: Number of commodities (default: 3)
- `density`: Probability of an edge existing (default: 0.3)
- `seed`: Seed for reproducibility (default: 42)
- `max_cap`: Maximum arc capacity (default: 100.0)
- `max_cost`: Maximum cost per arc (default: 20.0)
- `max_demand`: Maximum demand per commodity (default: 10.0)

### Solving

- `max_iterations`: Maximum number of iterations (default: 10000)
- `verbose`: Display detailed log (default: true)
- `tolerance`: Tolerance for reduced costs (default: 1e-6)

## 📝 Complete Example

```julia
# 1. Generate instance
instance = generate_mcnfp_instance(
    num_nodes=8,
    num_commodities=4,
    density=0.4,
    seed=123
)

# 2. Check feasibility
is_possible = check_feasibility_lp(instance)

# 3. Solve
if is_possible
    result = solve_mcnfp_column_generation(
        instance;
        max_iterations=1000,
        verbose=true
    )
    
    println("Result: ", result.optimal ? "Optimal" : "Non-optimal")
    println("Total cost: ", result.objective_value)
end
```

## 🐛 Troubleshooting

### Infeasible Instance

If the instance is infeasible, the code:
- Uses slack variables to ensure model feasibility
- Tries to find alternative paths
- Generates diagnostics about overloaded arcs

### Slow Convergence

If the algorithm does not converge quickly:
- Increase `max_iterations`
- Check if the instance is feasible
- Consider increasing graph density or capacities

## 📚 References

This code was developed in the context of **Integer Programming** and **Mathematical Modeling** studies, implementing classical network optimization techniques.

## 📄 License

This code is provided for educational and research purposes.

## 👤 Author

Developed as part of a Doctor's degree in Mathematical Modeling.

---

**Note:** This code is a modified version that uses the Bellman-Ford algorithm when necessary, unlike previous versions that used only Dijkstra.
