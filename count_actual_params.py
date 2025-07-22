#!/usr/bin/env python3
"""
Count actual parameters and estimate FLOPs for the exact models being used
"""
import numpy as np

def manual_flop_analysis():
    """Manual FLOP analysis using the actual parameters from terminal output"""
    
    print("=== Actual Model Analysis from Terminal Output ===")
    print("Input shape: (40000, 256, 1)")
    print("UNet parameters: 170,457")
    print("Transformer parameters: 134,081")
    
    # Configuration from ks.yaml
    seq_len = 256
    d_input = 1
    d_output = 1
    
    # Transformer config from ks.yaml
    num_layers = 4
    d_model = 64
    num_heads = 4
    dim_feedforward = 128
    
    print(f"\n=== Transformer FLOP Analysis ===")
    print(f"Config: {num_layers} layers, d_model={d_model}, heads={num_heads}, ff={dim_feedforward}")
    
    # Input projection: d_input * d_model * seq_len
    input_proj_flops = d_input * d_model * seq_len
    
    # Per layer calculations
    layer_flops = 0
    
    # Multi-head attention per layer
    # Q, K, V projections: 3 * (d_model * d_model * seq_len)
    qkv_proj_flops = 3 * d_model * d_model * seq_len
    
    # Attention computation (this is the killer!)
    # Q @ K^T: seq_len * seq_len * d_model  
    # Attention @ V: seq_len * seq_len * d_model
    # Plus softmax overhead: ~4 * seq_len * seq_len
    attention_flops = 2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len
    
    # Output projection: d_model * d_model * seq_len
    out_proj_flops = d_model * d_model * seq_len
    
    # Feed-forward network per layer
    # Linear 1: d_model * dim_feedforward * seq_len
    # Linear 2: dim_feedforward * d_model * seq_len  
    ff_flops = d_model * dim_feedforward * seq_len + dim_feedforward * d_model * seq_len
    
    # Layer norms: ~4 ops per element, 2 layer norms per layer
    ln_flops = 2 * 4 * d_model * seq_len
    
    layer_flops = qkv_proj_flops + attention_flops + out_proj_flops + ff_flops + ln_flops
    total_transformer_layers_flops = layer_flops * num_layers
    
    # Output projection
    output_proj_flops = d_model * d_output * seq_len
    
    total_transformer_flops = input_proj_flops + total_transformer_layers_flops + output_proj_flops
    
    print(f"Input projection: {input_proj_flops:,} FLOPs")
    print(f"Per layer breakdown:")
    print(f"  QKV projections: {qkv_proj_flops:,}")
    print(f"  Attention computation: {attention_flops:,}")
    print(f"  Output projection: {out_proj_flops:,}")
    print(f"  Feed-forward: {ff_flops:,}")
    print(f"  Layer norms: {ln_flops:,}")
    print(f"  Total per layer: {layer_flops:,}")
    print(f"All {num_layers} layers: {total_transformer_layers_flops:,}")
    print(f"Output projection: {output_proj_flops:,}")
    print(f"TOTAL Transformer FLOPs: {total_transformer_flops:,}")
    
    # UNet estimation (rough - need to verify against actual architecture)
    print(f"\n=== UNet FLOP Analysis (Estimated) ===")
    print("Note: UNet architecture more complex, this is a rough estimate")
    
    # For 1D UNet with 170k parameters, rough estimate:
    # Encoder: 5 levels, decoder: 4 levels, various channel sizes
    # This is much harder to estimate without exact architecture details
    
    # Very rough estimate based on typical 1D UNet operations
    # Assuming ~50-100 FLOPs per parameter for conv operations
    unet_flops_estimate = 170457 * 75  # rough multiplier
    
    print(f"Estimated UNet FLOPs: {unet_flops_estimate:,}")
    
    print(f"\n=== COMPARISON ===")
    print(f"Transformer FLOPs: {total_transformer_flops:,}")
    print(f"UNet FLOPs (est): {unet_flops_estimate:,}")
    print(f"Transformer is ~{total_transformer_flops / unet_flops_estimate:.1f}x more expensive")
    
    print(f"\nParameters:")
    print(f"Transformer: 134,081")
    print(f"UNet: 170,457")
    print(f"UNet has {170457/134081:.1f}x more parameters")
    
    print(f"\nFLOPs per parameter:")
    print(f"Transformer: {total_transformer_flops/134081:.1f}")
    print(f"UNet (est): {unet_flops_estimate/170457:.1f}")
    
    print(f"\n=== KEY INSIGHT: WHY TRANSFORMER IS SLOW ===")
    print(f"The attention mechanism is O(n²) with sequence length!")
    print(f"Sequence length: {seq_len}")
    print(f"Attention FLOPs per layer: {attention_flops:,}")
    print(f"Total attention FLOPs: {attention_flops * num_layers:,}")
    print(f"Attention is {(attention_flops * num_layers)/total_transformer_flops:.1%} of total FLOPs")
    
    print(f"\nScaling nightmare:")
    for scale in [0.5, 2, 4]:
        scaled_len = int(seq_len * scale)
        scaled_attention = 2 * scaled_len * scaled_len * d_model + 4 * scaled_len * scaled_len
        print(f"  Seq length {scaled_len}: attention = {scaled_attention:,} FLOPs ({scale**2:.1f}x scaling)")
    
    # The quadratic scaling is the killer!
    attention_component = attention_flops * num_layers
    print(f"\nWith seq_len=256, attention alone needs {attention_component:,} FLOPs")
    print(f"That's {attention_component/(seq_len**2):.1f} FLOPs per sequence position squared!")

if __name__ == "__main__":
    manual_flop_analysis()
