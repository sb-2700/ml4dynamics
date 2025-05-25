# Active Context

## Current Work Focus
The current focus is on enhancing the UNet implementation in the models module to support configurable kernel sizes. This change will improve the flexibility of the model architecture, allowing experimentation with different kernel sizes for various simulation tasks.

## Recent Changes

### UNet Enhancement with Configurable Kernel Sizes
- Added `kernel_size` parameter to the UNet class and its components (Encoder1D, Decoder1D, Encoder2D, Decoder2D)
- Modified all convolutional layers to use the configurable kernel size
- Maintained specialized kernel sizes for upsampling (2,2) and final output projection (1,1)
- Ensured proper parameter passing between UNet and its sub-components

### Bugfix in NS Equations Crank-Nicolson Implementation
- Identified and fixed inconsistency in the velocity calculation from vorticity in the NS spectral solver
- Corrected implementation of nonlinear term calculation

## Next Steps

### Near-term Tasks
1. **Testing of UNet with Different Kernel Sizes**: 
   - Evaluate model performance with varying kernel sizes
   - Document optimal kernel sizes for different types of simulation data

2. **Numerical Improvements**:
   - Implement dealiasing for spectral methods
   - Add JIT compilation to performance-critical sections
   - Improve error handling for numerical instabilities

3. **Documentation**:
   - Add docstrings to key functions
   - Document mathematical formulations of numerical methods

### Medium-term Goals
1. **Performance Optimization**:
   - Profile and optimize bottlenecks in both simulation and ML code
   - Explore parallel implementations for data generation

2. **Model Architecture Exploration**:
   - Experiment with additional neural network architectures beyond UNet
   - Implement attention mechanisms for improved feature extraction

3. **Advanced Regularization Techniques**:
   - Implement and evaluate additional regularization methods for distribution shift mitigation
   - Compare tangent-space regularization with other approaches

## Active Decisions and Considerations

### Numerical Stability vs. Performance
- **Decision Point**: How to balance numerical stability with computational performance
- **Current Approach**: Using double precision (float64) for critical numerical operations
- **Consideration**: Exploring mixed precision approaches for ML components

### Modularization and Code Organization
- **Decision Point**: How to refactor large files like utils.py and dynamics.py
- **Current Approach**: Planning modular structure while maintaining current functionality
- **Consideration**: Breaking down by functionality vs. by simulation type

### ML Framework Selection
- **Decision Point**: When to use JAX vs. PyTorch for different components
- **Current Approach**: JAX for performance-critical simulation code, flexibility in ML frameworks
- **Consideration**: Standardizing on JAX for better integration while maintaining PyTorch implementations for comparison 