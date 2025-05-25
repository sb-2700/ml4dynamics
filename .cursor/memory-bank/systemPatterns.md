# System Patterns: NS Simulation Architecture

## System Architecture
The NS simulation system follows a modular design with clear separation of concerns:

1. Core Simulation Components:
   - `ns_hit` class: Main simulation class implementing the NS equations
   - Spectral solver: Handles spatial discretization
   - Time integrator: Manages temporal evolution

2. Data Flow:
   ```
   Initial Conditions → Spectral Transform → Time Integration → Physical Space
   ```

## Design Patterns

### 1. Spectral Method Implementation
- Uses FFT for spatial discretization
- Implements anti-aliasing through expansion method
- Maintains spectral accuracy through careful handling of nonlinear terms

### 2. Time Integration Pattern
- Crank-Nicolson scheme for stability
- Implicit treatment of linear terms
- Explicit treatment of nonlinear terms
- Modular design allowing for different time integration schemes

### 3. Boundary Condition Handling
- Periodic boundary conditions in spectral space
- Automatic enforcement through FFT properties
- No explicit boundary condition implementation needed

## Component Relationships

### 1. Main Classes
- `dynamics`: Base class for all dynamical systems
- `ns_hit`: Specialized class for NS equations
- Utility classes for data handling and visualization

### 2. Data Structures
- Spectral space arrays: Complex-valued for efficiency
- Physical space arrays: Real-valued for visualization
- Configuration objects: YAML-based parameter management

### 3. Key Functions
- `CN`: Time integration
- `assembly_spectral`: Spectral space setup
- `set_x_hist`: History tracking
- `calc_J`: Nonlinear term calculation

## Implementation Patterns

### 1. Numerical Methods
- Spectral accuracy through FFT
- Implicit-explicit time splitting
- Anti-aliasing through expansion

### 2. Performance Optimizations
- JAX JIT compilation
- Vectorized operations
- Efficient memory management

### 3. Error Handling
- Stability checks
- Numerical validation
- Error tracking and reporting 