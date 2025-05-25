# Technical Context

## Technologies Used

### Core Libraries
- **JAX**: Primary framework for numerical computing and neural networks
- **Flax**: Neural network library built on JAX
- **NumPy/JAX NumPy**: Array manipulation and numerical operations
- **SciPy**: Scientific computing utilities
- **PyTorch**: Alternative implementation of some models
- **Matplotlib**: Visualization and plotting

### Development Tools
- **Python 3.9+**: Primary programming language
- **Conda**: Environment management
- **Git**: Version control
- **pytest**: Testing framework

## Development Setup

### Environment Setup
```bash
conda create -n TR python=3.9.18
conda activate TR
python -m pip install -r requirements.txt
```

### Project Structure
```
ml4dynamics/
├── __init__.py
├── dynamics.py
├── utils.py
├── types.py
├── models/
├── dataset_utils/
├── exp/
├── trainers/
├── tests/
├── config/
├── data/
│   ├── NS/
│   └── RD/
├── results/
│   └── fig/
└── ckpts/
```

### Data Flow
1. **Data Generation**: `generate_data.sh` script creates data in the data/ directory
2. **Model Training**: `train.sh` script trains models and saves checkpoints in ckpts/
3. **Evaluation**: Scripts in exp/ evaluate models and generate plots in results/fig/

## Technical Constraints

### Numerical Stability
- The Crank-Nicolson scheme requires careful implementation to maintain stability
- Spectral methods can suffer from aliasing issues without proper dealiasing
- Neural networks must be constrained to produce physically plausible outputs

### Computational Resources
- High-resolution simulations require significant memory and compute
- Training neural networks requires GPU acceleration
- Some operations are memory-bound rather than compute-bound

### JAX Limitations
- JAX's pure functional approach requires different coding patterns from imperative code
- Dynamic shapes are challenging in a JIT-compiled context
- In-place updates are simulated, not actual in-place operations

## Dependencies

### Main Dependencies (from requirements.txt)
- JAX and JAXLIB (with CUDA support)
- Flax
- Optax (for optimization)
- PyTorch
- Matplotlib
- NumPy
- SciPy
- h5py (for data storage)
- ml_collections (for configuration)
- python-box (for configuration)

### Optional Dependencies
- CUDA libraries for GPU acceleration
- Jupyter for interactive development and visualization

## Deployment

### Typical Workflow
1. **Setup**: Install dependencies and prepare environment
2. **Data Generation**: Generate simulation data for training
3. **Training**: Train ML models on simulation data
4. **Evaluation**: Evaluate models in various simulation scenarios
5. **Analysis**: Analyze results and generate plots/visualizations

### Performance Considerations
- Use JIT compilation for performance-critical sections
- Vectorize operations wherever possible
- Use GPU acceleration for both simulations and ML training
- Balance memory usage and computational performance
- Use appropriate precision (float64 is often needed for scientific computing) 