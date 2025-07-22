#!/usr/bin/env python3
"""
Count FLOPs for different model architectures using exact configurations
"""
import jax
import jax.numpy as jnp
import yaml
from box import Box
from flax import linen as nn
from ml4dynamics.models.models_jax import UNet, Transformer1D

def get_actual_configs():
    """Get the actual configurations used in your project"""
    # Load config
    with open('config/ks.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Based on your mention: inputs shape: (40000, 256, 1), total parameters for ols_transformer: 134081
    actual_input_shape = (40000, 256, 1)  # You specified this shape
    
    # Transformer config from ks.yaml
    transformer_config = {
        'num_layers': config['model']['num_layers'],  # 4
        'd_model': config['model']['d_model'],        # 64  
        'num_heads': config['model']['num_heads'],    # 4
        'dim_feedforward': config['model']['dim_feedforward'],  # 128
        'dropout_prob': config['model']['dropout_prob'],  # 0.1
        'max_len': config['model']['max_len'],         # 2048
        'input_dim': 1,  # from your shape (40000, 256, 1)
        'output_dim': 1
    }
    
    # UNet config - using same input shape for comparison  
    unet_config = {
        'input_features': 1,  # from your shape  
        'output_features': 1,
        'DIM': 1,
        'kernel_size': config['train']['kernel_size'],  # 3
        'training': False
    }
    
    return actual_input_shape, transformer_config, unet_config

def count_flops_unet(input_shape, unet_config):
    """Count FLOPs for UNet model with actual config"""
    print("=== UNet FLOP Analysis (Actual Config) ===")
    
    # Create UNet model with actual parameters
    model = UNet(
        input_features=unet_config['input_features'],
        output_features=unet_config['output_features'], 
        DIM=unet_config['DIM'],
        kernel_size=unet_config['kernel_size'],
        dtype=jnp.float32,
        training=unet_config['training']
    )
    
    
    # Initialize parameters - use single sample for analysis
    key = jax.random.PRNGKey(0)
    # Use single sample: (1, 256, 1) instead of full batch
    single_sample_shape = (1, input_shape[1], input_shape[2])
    x = jnp.ones(single_sample_shape)
    params = model.init(key, x)
    
    # Count parameters
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")
    
    # Manual FLOP estimation for UNet
    batch_size, seq_len, channels = single_sample_shape
    
    # Encoder blocks (5 levels with pooling)
    encoder_flops = 0
    current_len = seq_len
    current_channels = channels
    
    for level in range(5):
        features = (unet_config['input_features'] * 8) * (2 ** level)  # Use actual input_features
        
        # Two conv layers per block
        for _ in range(2):
            # Conv1D: kernel_size * in_channels * out_channels * output_length
            conv_flops = unet_config['kernel_size'] * current_channels * features * current_len
            encoder_flops += conv_flops
            current_channels = features
        
        # BatchNorm: ~4 ops per element 
        bn_flops = 4 * features * current_len
        encoder_flops += bn_flops
        
        # Pooling (except last level): reduces length by 2
        if level < 4:
            current_len = current_len // 2
    
    # Decoder blocks (4 levels with upsampling)
    decoder_flops = 0
    
    for level in range(4):
        features = (unet_config['input_features'] * 8) * (2 ** (3 - level))
        
        # Upsampling: minimal ops
        current_len *= 2
        
        # Conv transpose: similar to conv
        conv_flops = 2 * features * features * current_len
        decoder_flops += conv_flops
        
        # Two conv layers per block
        for _ in range(2):
            conv_flops = unet_config['kernel_size'] * features * features * current_len
            decoder_flops += conv_flops
        
        # BatchNorm
        bn_flops = 4 * features * current_len
        decoder_flops += bn_flops
    
    # Final conv layer
    final_conv_flops = 1 * (unet_config['input_features'] * 8) * unet_config['output_features'] * seq_len
    
    total_flops = encoder_flops + decoder_flops + final_conv_flops
    
    print(f"Encoder FLOPs: {encoder_flops:,}")
    print(f"Decoder FLOPs: {decoder_flops:,}")
    print(f"Final conv FLOPs: {final_conv_flops:,}")
    print(f"Total estimated FLOPs: {total_flops:,}")
    print(f"FLOPs per parameter: {total_flops / total_params:.1f}")
    
    return total_flops, total_params

def count_flops_transformer(input_shape, transformer_config):
    """Count FLOPs for Transformer model with actual config"""
    print("\n=== Transformer FLOP Analysis (Actual Config) ===")
    
    # Create transformer with actual config
    model = Transformer1D(
        num_layers=transformer_config['num_layers'],
        input_dim=transformer_config['input_dim'],
        output_dim=transformer_config['output_dim'],
        d_model=transformer_config['d_model'],
        num_heads=transformer_config['num_heads'],
        dim_feedforward=transformer_config['dim_feedforward'],
        dropout_prob=transformer_config['dropout_prob'],
        max_len=transformer_config['max_len']
    )
    
    # Initialize parameters - use single sample for analysis
    key = jax.random.PRNGKey(0)
    # Use single sample: (1, 256, 1) instead of full batch
    single_sample_shape = (1, input_shape[1], input_shape[2])
    x = jnp.ones(single_sample_shape)
    params = model.init(key, x, train=False)
    
    # Count parameters
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")
    print(f"Expected parameters: 134,081 (you mentioned)")
    
    # Manual FLOP estimation for Transformer
    batch_size, seq_len, d_input = single_sample_shape
    num_layers = transformer_config['num_layers']
    d_model = transformer_config['d_model']
    num_heads = transformer_config['num_heads']
    d_ff = transformer_config['dim_feedforward']
    
    # Input projection
    input_proj_flops = d_input * d_model * seq_len
    
    # Transformer layers
    transformer_flops = 0
    
    for layer in range(num_layers):
        # Multi-head attention
        # Q, K, V projections: 3 * (d_model * d_model * seq_len)
        qkv_proj_flops = 3 * d_model * d_model * seq_len
        
        # Attention computation: 
        # Q @ K^T: seq_len * seq_len * d_model
        # Softmax: ~4 * seq_len * seq_len (exp, sum, div)
        # Attention @ V: seq_len * seq_len * d_model
        attention_flops = 2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len
        
        # Output projection: d_model * d_model * seq_len
        out_proj_flops = d_model * d_model * seq_len
        
        # Feed-forward network:
        # Linear 1: d_model * d_ff * seq_len
        # ReLU: seq_len * d_ff (minimal)
        # Linear 2: d_ff * d_model * seq_len
        ff_flops = d_model * d_ff * seq_len + d_ff * d_model * seq_len
        
        # Layer norms: ~4 ops per element
        ln_flops = 2 * 4 * d_model * seq_len
        
        layer_flops = qkv_proj_flops + attention_flops + out_proj_flops + ff_flops + ln_flops
        transformer_flops += layer_flops
    
    # Output projection
    output_proj_flops = d_model * transformer_config['output_dim'] * seq_len
    
    total_flops = input_proj_flops + transformer_flops + output_proj_flops
    
    print(f"Input projection FLOPs: {input_proj_flops:,}")
    print(f"Transformer layers FLOPs: {transformer_flops:,}")
    print(f"Output projection FLOPs: {output_proj_flops:,}")
    print(f"Total estimated FLOPs: {total_flops:,}")
    print(f"FLOPs per parameter: {total_flops / total_params:.1f}")
    
    # Breakdown per layer
    avg_layer_flops = transformer_flops // num_layers if num_layers > 0 else 0
    print(f"\nPer layer breakdown:")
    print(f"  QKV projections: {3 * d_model * d_model * seq_len:,}")
    print(f"  Attention computation: {2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len:,}")
    print(f"  Output projection: {d_model * d_model * seq_len:,}")
    print(f"  Feed-forward: {2 * d_model * d_ff * seq_len:,}")
    print(f"  Layer norms: {2 * 4 * d_model * seq_len:,}")
    print(f"  Total per layer: {avg_layer_flops:,}")
    
    # Key insight: Attention is O(n^2) with sequence length!
    attention_fraction = (2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len) / (total_flops / num_layers)
    print(f"  Attention is {attention_fraction:.1%} of each layer's FLOPs")
    
    return total_flops, total_params

def compare_models():
    """Compare both models with actual configurations"""
    # Get actual configurations
    input_shape, transformer_config, unet_config = get_actual_configs()
    
    print("Comparing model computational complexity...")
    print(f"Input shape: {input_shape} (batch, sequence, features)")
    print(f"Single forward pass shape: (1, {input_shape[1]}, {input_shape[2]})\n")
    
    unet_flops, unet_params = count_flops_unet(input_shape, unet_config)
    transformer_flops, transformer_params = count_flops_transformer(input_shape, transformer_config)
    
    print(f"\n=== COMPARISON ===")
    print(f"UNet FLOPs: {unet_flops:,}")
    print(f"Transformer FLOPs: {transformer_flops:,}")
    print(f"Transformer is {transformer_flops / unet_flops:.1f}x more expensive")
    
    print(f"\nUNet parameters: {unet_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    print(f"Transformer has {transformer_params / unet_params:.1f}x more parameters")
    
    print(f"\nFLOPs per parameter:")
    print(f"UNet: {unet_flops / unet_params:.1f}")
    print(f"Transformer: {transformer_flops / transformer_params:.1f}")
    
    print(f"\n=== KEY INSIGHTS ===")
    seq_len = input_shape[1]
    d_model = transformer_config['d_model']
    
    # Attention complexity analysis
    attention_flops_per_layer = 2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len
    total_attention_flops = attention_flops_per_layer * transformer_config['num_layers']
    
    print(f"Transformer attention is O(n²) with sequence length!")
    print(f"Sequence length: {seq_len}")
    print(f"Attention FLOPs per layer: {attention_flops_per_layer:,}")
    print(f"Total attention FLOPs: {total_attention_flops:,}")
    print(f"Attention is {total_attention_flops / transformer_flops:.1%} of total Transformer FLOPs")
    
    # Scaling analysis
    print(f"\nScaling with sequence length:")
    for scale in [0.5, 2, 4]:
        scaled_len = int(seq_len * scale)
        scaled_attention = 2 * scaled_len * scaled_len * d_model + 4 * scaled_len * scaled_len
        print(f"  Length {scaled_len}: attention FLOPs = {scaled_attention:,} ({scale**2:.1f}x)")

if __name__ == "__main__":
    compare_models()
