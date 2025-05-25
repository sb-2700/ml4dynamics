# Progress

## Current Status
The project is in active development with core dynamics simulations and machine learning components implemented. Recent work has focused on enhancing model flexibility and fixing numerical stability issues.

## What Works

### Dynamics Simulations
- ✅ Lorenz system simulation
- ✅ Rossler system simulation
- ✅ Kuramoto-Sivashinsky (KS) equation simulation
- ✅ 2D Navier-Stokes simulation for homogeneous isotropic turbulence (NS-HIT)
- ✅ 2D Navier-Stokes simulation for channel flow
- ✅ Reaction-diffusion system simulation

### Machine Learning Components
- ✅ JAX-based UNet implementation (1D and 2D)
- ✅ PyTorch-based UNet implementation
- ✅ MLP implementation
- ✅ cVAE implementation
- ✅ Training pipeline with customizable optimization

### Numerical Methods
- ✅ Crank-Nicolson scheme for PDEs
- ✅ Spectral methods for spatial discretization
- ✅ Finite difference methods

### Utilities
- ✅ Data loading and preprocessing
- ✅ Basic visualization tools
- ✅ Experiment tracking
- ✅ Model evaluation metrics

## What's Left to Build

### Dynamics Simulations
- ⬜ Higher-order time integration methods
- ⬜ 3D Navier-Stokes simulation
- ⬜ More complex reaction-diffusion systems
- ⬜ Chaotic systems library expansion

### Machine Learning Components
- ⬜ Attention-based architectures
- ⬜ Physics-informed neural networks (PINNs)
- ⬜ Transfer learning capabilities
- ⬜ Uncertainty quantification

### Numerical Methods
- ⬜ Adaptive time-stepping
- ⬜ Multi-grid methods
- ⬜ Dealiasing for spectral methods

### Utilities
- ⬜ Advanced visualization tools
- ⬜ Comprehensive documentation
- ⬜ Extended test coverage
- ⬜ Performance benchmarking suite

## Recent Accomplishments
1. Added configurable kernel sizes to UNet implementation
2. Fixed bugs in the Navier-Stokes Crank-Nicolson implementation
3. Implemented initial framework for tangent-space regularization
4. Enhanced data processing pipeline for simulation data

## Known Issues

### Numerical Stability
- ⚠️ Crank-Nicolson scheme can be unstable for large time steps in some configurations
- ⚠️ Spectral methods suffer from aliasing in high-wavenumber regimes
- ⚠️ Nonlinear terms in fluid simulations can lead to instabilities

### Performance
- ⚠️ Large simulation data can cause memory bottlenecks
- ⚠️ Some routines are not optimally vectorized
- ⚠️ JIT compilation overhead can be significant for smaller problems

### Code Organization
- ⚠️ utils.py and dynamics.py are too large and need refactoring
- ⚠️ Inconsistent API design across different simulation types
- ⚠️ Limited documentation for complex numerical methods

## Next Milestone
Complete the implementation of dealiasing for spectral methods and test its impact on simulation stability, particularly for the Navier-Stokes equations. 