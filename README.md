# ML for Combinatorial Optimization: Edge Coloring Framework

This project provides a comprehensive framework that integrates machine learning models with traditional combinatorial optimization techniques to solve edge coloring problems in graphs.

## Overview

Edge coloring is the task of assigning colors to graph edges such that no two adjacent edges share the same color. This framework leverages multiple approaches including:
- **Greedy Algorithms & Local Search:** Generate initial coloring solutions.
- **Machine Learning Models:** Train and evaluate Random Forests, Graph Neural Networks, and hybrid approaches to predict or refine optimal colorings.
- **Graph Generation & Feature Extraction:** Create diverse graph instances and extract graph-level and edge-level features for model training.
- **Visualization:** Compare performance metrics, visualize feature importance, and demonstrate solution quality.

## Key Features

- **Multiple Graph Types:** Generate and analyze random, scale-free, small-world, and geometric graphs
- **Diverse Coloring Algorithms:** Implement greedy, Vizing, ILP, and local search approaches
- **Advanced ML Models:** Train Random Forests, Graph Neural Networks, and hybrid models
- **Comprehensive Feature Engineering:** Extract both graph-level and edge-level features
- **Flexible Pipeline:** Run the complete workflow or individual components
- **Extensive Visualization:** Compare algorithm performance and analyze results

## Directory Structure

```
ML-for-Combinatorial-Optimization/
├── data/
│   ├── raw/                # Generated graphs and colorings
│   ├── processed/          # Extracted features and datasets
│   └── results/            # Evaluation outputs and visualizations
├── src/
│   ├── graph_generation/   # Scripts for generating various graph types
│   ├── coloring/           # Edge coloring algorithms (greedy, Vizing, ILP, local search)
│   ├── features/           # Modules for extracting graph and edge features
│   ├── models/             # Implementations of Random Forest, GNN, and Hybrid models
│   ├── training/           # Dataset preparation, training, and evaluation routines
│   ├── visualization/      # Code to visualize graphs and evaluation results
│   └── utils.py            # Utility functions used across the project
├── notebooks/              # Jupyter notebooks for data exploration and case studies
├── config.py               # Configuration settings and parameters
├── main.py                 # Main entry point to run the complete pipeline
└── requirements.txt        # Project dependencies
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd ML-for-Combinatorial-Optimization
   ```

2. **Create and Activate a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Note: For ILP solver functionality, you'll need to install additional packages:
   - PuLP (`pip install pulp`) for open-source solver
   - OR CPLEX (requires IBM CPLEX installation)

## Usage

The framework is executed through `main.py`, which supports various command-line arguments to run specific parts of the pipeline:

```bash
python main.py [OPTIONS]
```

### Common Options

- `--generate-graphs`: Generate graph instances
- `--generate-colorings`: Generate edge colorings for graphs
- `--extract-features`: Extract features from graphs and colorings
- `--prepare-datasets`: Prepare training datasets
- `--train-models`: Train the machine learning models
- `--evaluate-models`: Evaluate the trained models
- `--visualize-results`: Visualize evaluation and model performance
- `--run-all`: Run the complete pipeline

### Additional Parameters

- `--graph-types`: Types of graphs to generate (random, scale_free, small_world, geometric)
- `--coloring-methods`: Methods to use for generating edge colorings
- `--models`: Machine learning models to train and evaluate
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable verbose output

### Examples

Run the complete pipeline:
```bash
python main.py --run-all
```

Generate only graph instances:
```bash
python main.py --generate-graphs --graph-types random scale_free --num-per-config 10
```

Train and evaluate specific models:
```bash
python main.py --train-models --evaluate-models --models random_forest gnn
```

Generate visualizations of results:
```bash
python main.py --visualize-results
```

## Experimental Results

The framework has been tested on various graph types and sizes, with the following key findings:

- Hybrid models combining ML with local search achieve the best solution quality, approaching the theoretical bounds in many cases
- GNN models perform particularly well on scale-free networks, where structural patterns are more pronounced
- Random Forests provide good performance with significantly lower computational requirements
- Feature importance analysis reveals that maximum degree, edge density, and degree variance are the most influential graph characteristics for edge coloring

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds on research in machine learning for combinatorial optimization
- Special thanks to all contributors and advisors involved in this project
- The implementation uses several open-source libraries including NetworkX, PyTorch Geometric, and scikit-learn