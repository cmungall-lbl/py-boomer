# BOOMER-PY

BOOMER-PY (Bayesian OWL Ontology MErgER in Python) is a probabilistic reasoning system for knowledge representation and ontological reasoning with uncertainty.

## Overview

BOOMER-PY enables reasoning over probabilistic facts and taxonomic relationships, finding the most likely consistent interpretation of potentially conflicting assertions. It uses a combination of graph-based reasoning and Bayesian probabilistic inference.

Key features:
- Represent probabilistic ontological statements
- Reason over class subsumption hierarchies
- Evaluate class equivalence relationships
- Detect and resolve logical inconsistencies
- Calculate posterior probabilities for each assertion
- Grid search with consensus solutions across parameter configurations
- Automatic graph partitioning for scalable reasoning

## Core Concepts

- **Knowledge Base (KB)**: Collection of facts and probabilistic facts (PFacts)
- **Facts**: Logical assertions about entity relationships
  - SubClassOf: A is a subclass of B
  - ProperSubClassOf: A is a proper subclass of B (A � B)
  - EquivalentTo: A is equivalent to B
  - NotInSubsumptionWith: A is not in a subsumption relationship with B
  - MemberOfDisjointGroup: A belongs to disjoint group G
- **Probabilistic Facts**: Facts with assigned probabilities
- **Reasoning**: Logical deduction over facts to find satisfiable solutions
- **Search**: Exploration of possible combinations of assertions

## Use Cases

BOOMER-PY is designed for:
- Merging ontologies with uncertain mapping relationships
- Reasoning with probabilistic taxonomies
- Resolving conflicts in knowledge bases
- Scientific knowledge representation with uncertainty

## Example

```python
from boomer.model import KB, PFact, EquivalentTo
from boomer.search import solve
from boomer.renderers.markdown_renderer import MarkdownRenderer

# Create a knowledge base with probabilistic facts
kb = KB(
    pfacts=[
        PFact(EquivalentTo("cat", "Felix"), 0.9),
        PFact(EquivalentTo("dog", "Canus"), 0.9),
        PFact(EquivalentTo("cat", "Canus"), 0.1),
    ]
)

# Solve to find most probable consistent solution
solution = solve(kb)

# Display results
renderer = MarkdownRenderer()
print(renderer.render(solution))
```

## Grid Search and Consensus Solutions

BOOMER-PY supports grid search over hyperparameters with automatic aggregation and consensus building:

```python
from boomer.model import KB, SearchConfig, GridSearch
from boomer.search import grid_search

# Define parameter grid
grid = GridSearch(
    configurations=[
        SearchConfig(max_iterations=1000),
        SearchConfig(max_iterations=5000),
    ],
    configuration_matrix={
        "max_pfacts_per_clique": [20, 50, 100],
        "partition_initial_threshold": [30, 60],
    }
)

# Run grid search (expands to 2×3×2 = 12 configurations)
results = grid_search(kb, grid, eval_kb=ground_truth_kb)

# Access synthesized consensus solution
consensus = results.synthesized_solution
print(f"High confidence mappings: {len(consensus.high_confidence_facts)}")
print(f"Best config (by F1): {results.best_config}")

# Aggregate statistics across all configurations
stats = results.aggregate_stats
print(f"Mean F1: {stats.mean_f1:.2f} ± {stats.std_f1:.2f}")
print(f"Success rate: {stats.success_rate:.0%}")

# Pareto frontier for speed vs accuracy trade-off
for config in results.pareto_frontier:
    print(f"Config: {config.result.time_elapsed:.1f}s, F1: {config.evaluation.f1:.2f}")
```

The synthesized solution identifies mappings that are robustly accepted across parameter variations, providing higher confidence than single-configuration solutions.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/boomer-py.git
cd boomer-py

# Install dependencies
pip install .
```

## Development

BOOMER-PY uses:
- NetworkX for graph-based reasoning
- Pydantic for data modeling
- Pytest for testing

To run tests:
```bash
make test
```
