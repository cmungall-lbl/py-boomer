# Grid Search API Documentation

## Overview

The grid search functionality in Boomer-Py enables systematic exploration of hyperparameter spaces with automatic aggregation and consensus building. Rather than relying on a single configuration, grid search identifies robust mappings that are consistently accepted across parameter variations.

## Core Classes

### GridSearch

Main container for grid search configuration and results.

```python
class GridSearch(BaseModel):
    configurations: list[SearchConfig]
    configuration_matrix: dict[str, list[Any]] | None = None
    results: list[GridSearchResult] | None = None
    aggregate_stats: AggregateStats | None = None
    synthesized_solution: SynthesizedSolution | None = None
    best_config: SearchConfig | None = None
    best_config_metric: str | None = None
    pareto_frontier: List[GridSearchResult] | None = None
```

**Fields:**
- `configurations`: Base configurations to expand
- `configuration_matrix`: Parameter values for Cartesian product expansion
- `results`: Individual results for each configuration
- `aggregate_stats`: Summary statistics across all configurations
- `synthesized_solution`: Consensus solution via voting
- `best_config`: Optimal configuration by chosen metric
- `best_config_metric`: Metric used ("f1_score" or "confidence")
- `pareto_frontier`: Non-dominated configs in speed/accuracy space

### PFactConsensus

Tracks consensus for each probabilistic fact across configurations.

```python
class PFactConsensus(BaseModel):
    pfact: PFact
    acceptance_rate: float  # Proportion accepting this fact
    mean_posterior: float   # Mean posterior when accepted
    std_posterior: float    # Std dev of posterior
    consensus_score: float  # Weighted consensus (0-1)
    configurations_accepted: List[int]  # Config indices
    configurations_total: int
```

The consensus score combines acceptance rate with posterior probability strength, providing a single metric for mapping robustness.

### SynthesizedSolution

Aggregated solution combining evidence across all configurations.

```python
class SynthesizedSolution(BaseModel):
    pfact_consensus: List[PFactConsensus]
    aggregation_method: str = "weighted_vote"
    min_consensus_threshold: float = 0.5
    contributing_configs: int
    high_confidence_facts: List[PFact]  # >80% consensus
    uncertain_facts: List[PFact]        # 40-60% consensus
```

High confidence facts are those consistently accepted with strong posterior probabilities, while uncertain facts show parameter sensitivity.

### AggregateStats

Performance and quality metrics aggregated across configurations.

```python
class AggregateStats(BaseModel):
    # Performance metrics
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float

    # Solution quality
    mean_confidence: float
    std_confidence: float
    mean_posterior_prob: float

    # Computational metrics
    mean_time: float
    std_time: float
    mean_combinations_explored: int

    # Success metrics
    success_rate: float     # Configs finding solutions
    timeout_rate: float     # Configs timing out

    # Parameter impacts (optional)
    parameter_impacts: Dict[str, float] | None
```

## Functions

### grid_search()

Main entry point for grid search execution.

```python
def grid_search(
    kb: KB,
    grid: GridSearch,
    eval_kb: KB | None = None,
) -> GridSearch:
    """
    Perform grid search over hyperparameters.

    Args:
        kb: Knowledge base to solve
        grid: Grid search configuration
        eval_kb: Optional ground truth for evaluation

    Returns:
        GridSearch with populated results and aggregations
    """
```

The function:
1. Expands configuration matrix via Cartesian product
2. Runs solve() for each configuration
3. Optionally evaluates against ground truth
4. Computes aggregate statistics
5. Synthesizes consensus solution
6. Identifies best config and Pareto frontier

### compute_aggregate_stats()

Calculates summary statistics across all results.

```python
def compute_aggregate_stats(
    results: List[GridSearchResult]
) -> AggregateStats:
    """
    Compute mean, std dev, and rates across configurations.
    """
```

### synthesize_solution()

Creates consensus solution via weighted voting.

```python
def synthesize_solution(
    kb: KB,
    results: List[GridSearchResult]
) -> SynthesizedSolution:
    """
    Aggregate evidence across configurations.

    For each pfact:
    - Count configurations accepting it
    - Average posterior probabilities
    - Compute consensus score
    - Categorize by confidence level
    """
```

### find_pareto_frontier()

Identifies non-dominated configurations.

```python
def find_pareto_frontier(
    results: List[GridSearchResult]
) -> List[GridSearchResult]:
    """
    Find configs not dominated in speed/accuracy.

    Config A dominates B if:
    - A is faster AND more accurate
    """
```

## Usage Examples

### Basic Grid Search

```python
from boomer.model import KB, SearchConfig, GridSearch
from boomer.search import grid_search

# Define search space
grid = GridSearch(
    configurations=[
        SearchConfig(max_iterations=1000),
        SearchConfig(max_iterations=5000),
    ],
    configuration_matrix={
        "max_pfacts_per_clique": [10, 30, 50],
        "timeout_seconds": [30, 60],
    }
)

# Run search (2 × 3 × 2 = 12 configurations)
results = grid_search(kb, grid)

# Inspect results
print(f"Configurations tested: {len(results.results)}")
print(f"Mean F1: {results.aggregate_stats.mean_f1:.2f}")
print(f"Best config: {results.best_config}")
```

### With Ground Truth Evaluation

```python
# Create ground truth KB
eval_kb = KB(facts=[
    EquivalentTo(sub="A", equivalent="X"),
    EquivalentTo(sub="B", equivalent="Y"),
])

# Run with evaluation
results = grid_search(kb, grid, eval_kb=eval_kb)

# Results now include precision/recall/F1
for r in results.results:
    print(f"Config: {r.config.max_iterations}")
    print(f"  F1: {r.evaluation.f1:.2f}")
```

### Analyzing Consensus

```python
# Get synthesized solution
consensus = results.synthesized_solution

# Examine high-confidence mappings
for pfact in consensus.high_confidence_facts:
    print(f"Robust mapping: {pfact.fact}")

# Find parameter-sensitive mappings
for pc in consensus.pfact_consensus:
    if 0.4 <= pc.consensus_score <= 0.6:
        print(f"Uncertain: {pc.pfact.fact}")
        print(f"  Accepted by {pc.acceptance_rate:.0%} of configs")
        print(f"  Configs: {pc.configurations_accepted}")
```

### Speed vs Accuracy Trade-off

```python
# Examine Pareto frontier
for result in results.pareto_frontier:
    time = result.result.time_elapsed
    f1 = result.evaluation.f1 if result.evaluation else 0
    conf = result.result.confidence

    print(f"Pareto optimal config:")
    print(f"  Time: {time:.1f}s")
    print(f"  F1: {f1:.2f}, Confidence: {conf:.2f}")
    print(f"  Settings: {result.config}")
```

### Custom Aggregation

```python
# Access individual results for custom analysis
import numpy as np

times = [r.result.time_elapsed for r in results.results]
configs_under_10s = sum(1 for t in times if t < 10)

# Parameter impact analysis
iterations = [r.config.max_iterations for r in results.results]
f1_scores = [r.evaluation.f1 for r in results.results
             if r.evaluation]

correlation = np.corrcoef(iterations[:len(f1_scores)], f1_scores)[0,1]
print(f"Iteration-F1 correlation: {correlation:.2f}")
```

## Best Practices

### Configuration Design

1. **Start with coarse grid**: Begin with widely spaced parameter values
2. **Refine around optimum**: Once best region identified, search finer grid
3. **Include extremes**: Test both conservative and aggressive settings
4. **Orthogonal parameters**: Vary independent parameters for better coverage

### Interpreting Results

1. **Check aggregate stats first**: Look for high variance as instability indicator
2. **Examine consensus**: High-confidence facts are deployment-ready
3. **Review Pareto frontier**: Choose based on your speed/accuracy needs
4. **Investigate uncertain mappings**: May need manual review

### Performance Considerations

1. **Limit grid size**: Start with <20 configurations
2. **Use timeouts**: Prevent single configs from dominating runtime
3. **Partition early**: Lower thresholds reduce individual solve times
4. **Parallel potential**: Results can be computed independently

## Advanced Topics

### Weighted Consensus

The consensus score for each pfact is computed as:

```
consensus_score = acceptance_rate × mean_posterior_probability
```

This weights frequently accepted facts by their average confidence when accepted.

### Parameter Impact Analysis

Future enhancement to quantify each parameter's effect:

```python
parameter_impacts = {
    "max_iterations": 0.15,      # Low impact
    "max_pfacts_per_clique": 0.72,  # High impact
    "timeout_seconds": 0.03,     # Minimal impact
}
```

### Ensemble Methods

Synthesized solutions can be viewed as ensemble methods where each configuration votes on fact acceptance. This provides robustness similar to random forests in machine learning.

## See Also

- [SearchConfig Documentation](search_config.md) - Individual configuration options
- [Solution Documentation](solution.md) - Understanding solve() results
- [Evaluation Documentation](evaluation.md) - Ground truth comparison