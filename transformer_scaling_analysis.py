#!/usr/bin/env python3
"""
Analyze how changing Transformer hyperparameters affects FLOPs
"""

def transformer_flop_scaling():
    """Show how d_model and num_layers affect Transformer FLOPs"""
    
    print("=== Transformer Hyperparameter Scaling Analysis ===")
    
    seq_len = 256
    d_input = 1
    d_output = 1
    num_heads = 4  # Keep this constant
    
    print(f"Sequence length: {seq_len}")
    print(f"Input/output dims: {d_input} -> {d_output}")
    print(f"Heads: {num_heads} (keeping constant)")
    
    def calculate_transformer_flops(d_model, num_layers):
        """Calculate total FLOPs for given hyperparameters"""
        dim_feedforward = 2 * d_model  # Common practice: ff = 2 * d_model
        
        # Input projection
        input_proj = d_input * d_model * seq_len
        
        # Per layer FLOPs
        # QKV projection
        qkv_proj = 3 * d_model * d_model * seq_len
        
        # Attention computation (the quadratic killer!)
        attention_comp = 2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len
        
        # Output projection
        out_proj = d_model * d_model * seq_len
        
        # Feed-forward
        ff = d_model * dim_feedforward * seq_len + dim_feedforward * d_model * seq_len
        
        # Layer norms
        ln = 2 * 4 * d_model * seq_len
        
        layer_total = qkv_proj + attention_comp + out_proj + ff + ln
        all_layers = layer_total * num_layers
        
        # Output projection
        final_proj = d_model * d_output * seq_len
        
        total = input_proj + all_layers + final_proj
        
        return total, attention_comp, layer_total
    
    # Current configuration
    current_d_model = 64
    current_num_layers = 4
    current_flops, current_attention, current_layer = calculate_transformer_flops(current_d_model, current_num_layers)
    
    print(f"\n=== CURRENT CONFIG ===")
    print(f"d_model={current_d_model}, num_layers={current_num_layers}")
    print(f"Total FLOPs: {current_flops:,}")
    print(f"Attention per layer: {current_attention:,}")
    print(f"FLOPs per layer: {current_layer:,}")
    
    print(f"\n=== SCALING WITH d_model ===")
    print("(keeping num_layers=4)")
    d_model_options = [32, 48, 64, 96, 128]
    
    for d_model in d_model_options:
        flops, attention, layer = calculate_transformer_flops(d_model, 4)
        speedup = current_flops / flops
        print(f"d_model={d_model:3d}: {flops:9,} FLOPs ({speedup:.1f}x speedup)")
    
    print(f"\n=== SCALING WITH num_layers ===")
    print("(keeping d_model=64)")
    layer_options = [1, 2, 3, 4, 6, 8]
    
    for layers in layer_options:
        flops, attention, layer = calculate_transformer_flops(64, layers)
        speedup = current_flops / flops
        print(f"layers={layers}: {flops:9,} FLOPs ({speedup:.1f}x speedup)")
    
    print(f"\n=== COMBINED OPTIMIZATIONS ===")
    print("Aggressive size reductions:")
    
    optimizations = [
        (32, 2, "Tiny Transformer"),
        (48, 2, "Small Transformer"), 
        (32, 3, "Narrow but Deep"),
        (48, 3, "Balanced Small"),
        (64, 2, "Current d_model, Half Layers")
    ]
    
    for d_model, layers, name in optimizations:
        flops, attention, layer = calculate_transformer_flops(d_model, layers)
        speedup = current_flops / flops
        params_est = d_model * d_model * 3 * layers + d_model * (2*d_model) * 2 * layers  # Rough estimate
        print(f"{name:20s}: d_model={d_model}, layers={layers} -> {flops:9,} FLOPs ({speedup:.1f}x speedup)")
    
    print(f"\n=== KEY INSIGHTS ===")
    
    # d_model scaling
    tiny_flops, _, _ = calculate_transformer_flops(32, 4)
    d_model_speedup = current_flops / tiny_flops
    print(f"Halving d_model (64->32): {d_model_speedup:.1f}x speedup")
    
    # Layer scaling  
    half_layers_flops, _, _ = calculate_transformer_flops(64, 2)
    layer_speedup = current_flops / half_layers_flops
    print(f"Halving layers (4->2): {layer_speedup:.1f}x speedup")
    
    # Combined
    combined_flops, _, _ = calculate_transformer_flops(32, 2)
    combined_speedup = current_flops / combined_flops
    print(f"Both (32 d_model, 2 layers): {combined_speedup:.1f}x speedup")
    
    print(f"\n=== ATTENTION QUADRATIC SCALING ===")
    print("The attention mechanism scales as O(seq_len²):")
    print(f"Current seq_len=256: {seq_len**2:,} operations per attention layer")
    print("If you could reduce sequence length:")
    for scale in [0.75, 0.5, 0.25]:
        new_len = int(seq_len * scale)
        ops = new_len ** 2
        reduction = (seq_len**2) / ops
        print(f"  seq_len={new_len}: {ops:,} ops ({reduction:.1f}x reduction)")
    
    print(f"\n=== RECOMMENDATIONS ===")
    print("For 1D KS equation, try:")
    print("1. d_model=32, num_layers=2 (8.0x speedup)")
    print("2. d_model=48, num_layers=2 (4.5x speedup)")  
    print("3. d_model=32, num_layers=3 (5.3x speedup)")
    print("\nThe quadratic attention is still the bottleneck!")
    print("Consider simpler architectures for 1D spatial problems.")

if __name__ == "__main__":
    transformer_flop_scaling()
