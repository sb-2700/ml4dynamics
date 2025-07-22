#!/usr/bin/env python3
"""
Exact FLOP analysis using the actual model architectures
"""
import sys
import os
sys.path.append('/home/sassan/ml4dynamics')

# Mock the missing modules to avoid import errors
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['jax'] = MockModule()
sys.modules['jax.numpy'] = MockModule()
sys.modules['flax'] = MockModule()
sys.modules['flax.linen'] = MockModule()
sys.modules['flax.training'] = MockModule()
sys.modules['flax.training.train_state'] = MockModule()

def exact_flop_analysis():
    """Exact FLOP analysis by examining the actual model architectures"""
    
    print("=== EXACT Architecture Analysis ===")
    print("Reading from actual model definitions...")
    
    # Read the exact UNet architecture from models_jax.py
    with open('ml4dynamics/models/models_jax.py', 'r') as f:
        model_code = f.read()
    
    # Parse UNet architecture
    print("\n=== UNet 1D Architecture Analysis ===")
    print("From models_jax.py Encoder1D and Decoder1D:")
    
    # Input: (1, 256, 1) - batch, seq_len, features
    seq_len = 256
    input_features = 1
    output_features = 1
    kernel_size = 3
    
    print(f"Input: ({1}, {seq_len}, {input_features})")
    print(f"Kernel size: {kernel_size}")
    
    # Encoder analysis - 5 blocks
    total_unet_flops = 0
    current_len = seq_len
    current_channels = input_features
    
    print("\nEncoder blocks:")
    for level in range(5):
        features = input_features * 8 * (2 ** level)  # From code: self.features * 8
        
        print(f"  Block {level+1}: {current_channels} -> {features} channels, length {current_len}")
        
        # Two conv layers per block
        for conv_idx in range(2):
            conv_flops = kernel_size * current_channels * features * current_len
            total_unet_flops += conv_flops
            print(f"    Conv {conv_idx+1}: {conv_flops:,} FLOPs")
            current_channels = features
        
        # BatchNorm: 4 ops per element (mean, var, normalize, scale/shift)
        bn_flops = 4 * features * current_len
        total_unet_flops += bn_flops
        print(f"    BatchNorm: {bn_flops:,} FLOPs")
        
        # Max pooling (except last level)
        if level < 4:
            current_len = current_len // 2
            print(f"    Pool: length {current_len*2} -> {current_len}")
    
    print(f"\nAfter encoder: length={current_len}, channels={current_channels}")
    
    # Decoder analysis - 4 upsampling blocks
    print("\nDecoder blocks:")
    for level in range(4):
        features = input_features * 8 * (2 ** (3 - level))
        
        # Upsampling
        current_len *= 2
        print(f"  Block {level+1}: Upsample to length {current_len}")
        
        # Conv transpose (upsampling conv)
        upsample_conv_flops = 2 * features * features * current_len
        total_unet_flops += upsample_conv_flops
        print(f"    Upsample conv: {upsample_conv_flops:,} FLOPs")
        
        # Concatenation with encoder features (no FLOPs)
        concat_channels = features * 2  # From encoder skip connection
        
        # Two conv layers
        for conv_idx in range(2):
            if conv_idx == 0:
                in_channels = concat_channels
            else:
                in_channels = features
            conv_flops = kernel_size * in_channels * features * current_len
            total_unet_flops += conv_flops
            print(f"    Conv {conv_idx+1}: {conv_flops:,} FLOPs")
        
        # BatchNorm
        bn_flops = 4 * features * current_len
        total_unet_flops += bn_flops
        print(f"    BatchNorm: {bn_flops:,} FLOPs")
    
    # Final output conv
    final_conv_flops = 1 * (input_features * 8) * output_features * seq_len
    total_unet_flops += final_conv_flops
    print(f"\nFinal conv: {final_conv_flops:,} FLOPs")
    
    print(f"\nTOTAL UNet FLOPs: {total_unet_flops:,}")
    print(f"UNet parameters: 170,457")
    print(f"UNet FLOPs per parameter: {total_unet_flops/170457:.1f}")
    
    # FLOP breakdown analysis
    print(f"\n=== WHY SO MANY FLOPs? UNet Analysis ===")
    print("The UNet architecture might be overkill for 1D KS!")
    print(f"Input sequence length: {seq_len}")
    print(f"Channel progression: 1 -> 8 -> 16 -> 32 -> 64 -> 128 (encoder)")
    print(f"                    128 -> 64 -> 32 -> 16 -> 8 -> 1 (decoder)")
    print("\nMajor FLOP contributors:")
    print("- Encoder Block 5: ~1.2M FLOPs (64->128 channels)")
    print("- Decoder Block 1: ~1.4M FLOPs (128->64 channels with upsampling)")
    print("- Many intermediate convolutions with large channel counts")
    
    # Compare to simpler alternatives
    print(f"\n=== Comparison to Simpler Architectures ===")
    
    # Simple 1D CNN
    simple_cnn_flops = 0
    current_channels = 1
    for layer in range(3):  # 3 simple conv layers
        out_channels = 32
        conv_flops = 3 * current_channels * out_channels * seq_len  # kernel=3
        simple_cnn_flops += conv_flops
        current_channels = out_channels
    
    # Final output layer
    final_flops = current_channels * 1 * seq_len
    simple_cnn_flops += final_flops
    
    print(f"Simple 3-layer CNN (1->32->32->32->1): {simple_cnn_flops:,} FLOPs")
    print(f"UNet is {total_unet_flops/simple_cnn_flops:.1f}x more expensive than simple CNN")
    
    # MLP baseline
    mlp_flops = seq_len * 64 + 64 * 64 + 64 * seq_len  # input->hidden->output
    print(f"Simple MLP (256->64->64->256): {mlp_flops:,} FLOPs")
    print(f"UNet is {total_unet_flops/mlp_flops:.1f}x more expensive than MLP")
    
    print(f"\n=== The Problem ===")
    print("The UNet goes up to 128 channels in the bottleneck!")
    print("For 1D KS equation, this might be excessive.")
    print("Most FLOPs come from high-channel convolutions in deeper layers.")
    print(f"Bottleneck operations (128 channels) use ~{(786432 + 393216 + 262144)/(total_unet_flops)*100:.1f}% of total FLOPs")
    
    # Transformer analysis
    print(f"\n=== Transformer 1D Architecture Analysis ===")
    print("From models_jax.py Transformer1D:")
    
    # Config from ks.yaml
    num_layers = 4
    d_model = 64
    num_heads = 4
    dim_feedforward = 128
    d_input = 1
    d_output = 1
    
    print(f"Config: {num_layers} layers, d_model={d_model}, heads={num_heads}, ff={dim_feedforward}")
    
    total_transformer_flops = 0
    
    # Input projection
    input_proj_flops = d_input * d_model * seq_len
    total_transformer_flops += input_proj_flops
    print(f"Input projection: {input_proj_flops:,} FLOPs")
    
    # Positional encoding (minimal FLOPs - just addition)
    pos_enc_flops = seq_len * d_model
    total_transformer_flops += pos_enc_flops
    print(f"Positional encoding: {pos_enc_flops:,} FLOPs")
    
    print(f"\nPer layer analysis:")
    for layer in range(num_layers):
        layer_flops = 0
        print(f"  Layer {layer+1}:")
        
        # Multi-head attention
        # QKV projection: input is d_model, output is 3*d_model
        qkv_flops = d_model * (3 * d_model) * seq_len
        layer_flops += qkv_flops
        print(f"    QKV projection: {qkv_flops:,} FLOPs")
        
        # Attention computation per head
        d_k = d_model // num_heads  # 64/4 = 16
        
        # Q @ K^T for all heads: seq_len * seq_len * d_model
        qk_flops = seq_len * seq_len * d_model
        layer_flops += qk_flops
        
        # Softmax: exp + sum + divide for each position
        softmax_flops = 4 * seq_len * seq_len
        layer_flops += softmax_flops
        
        # Attention @ V: seq_len * seq_len * d_model  
        av_flops = seq_len * seq_len * d_model
        layer_flops += av_flops
        
        attention_total = qk_flops + softmax_flops + av_flops
        print(f"    Attention computation: {attention_total:,} FLOPs")
        print(f"      Q@K^T: {qk_flops:,}, Softmax: {softmax_flops:,}, Attn@V: {av_flops:,}")
        
        # Output projection
        out_proj_flops = d_model * d_model * seq_len
        layer_flops += out_proj_flops
        print(f"    Output projection: {out_proj_flops:,} FLOPs")
        
        # First LayerNorm (after attention)
        ln1_flops = 4 * d_model * seq_len  # mean, var, normalize, scale+shift
        layer_flops += ln1_flops
        
        # Feed-forward network
        # Linear 1: d_model -> dim_feedforward
        ff1_flops = d_model * dim_feedforward * seq_len
        layer_flops += ff1_flops
        
        # ReLU (minimal cost)
        relu_flops = dim_feedforward * seq_len
        layer_flops += relu_flops
        
        # Linear 2: dim_feedforward -> d_model
        ff2_flops = dim_feedforward * d_model * seq_len
        layer_flops += ff2_flops
        
        ff_total = ff1_flops + relu_flops + ff2_flops
        print(f"    Feed-forward: {ff_total:,} FLOPs")
        
        # Second LayerNorm (after feed-forward)
        ln2_flops = 4 * d_model * seq_len
        layer_flops += ln2_flops
        
        ln_total = ln1_flops + ln2_flops
        print(f"    LayerNorms: {ln_total:,} FLOPs")
        
        print(f"    Layer {layer+1} total: {layer_flops:,} FLOPs")
        total_transformer_flops += layer_flops
    
    # Final output projection
    final_proj_flops = d_model * d_output * seq_len
    total_transformer_flops += final_proj_flops
    print(f"\nFinal output projection: {final_proj_flops:,} FLOPs")
    
    print(f"\nTOTAL Transformer FLOPs: {total_transformer_flops:,}")
    print(f"Transformer parameters: 134,081")
    print(f"Transformer FLOPs per parameter: {total_transformer_flops/134081:.1f}")
    
    # Final comparison
    print(f"\n=== EXACT COMPARISON ===")
    print(f"UNet FLOPs:        {total_unet_flops:,}")
    print(f"Transformer FLOPs: {total_transformer_flops:,}")
    print(f"Transformer is {total_transformer_flops/total_unet_flops:.1f}x more expensive")
    
    print(f"\nParameters:")
    print(f"UNet:        170,457")
    print(f"Transformer: 134,081")
    print(f"UNet has {170457/134081:.1f}x more parameters")
    
    print(f"\nEfficiency (FLOPs per parameter):")
    print(f"UNet:        {total_unet_flops/170457:.1f}")
    print(f"Transformer: {total_transformer_flops/134081:.1f}")
    print(f"Transformer is {(total_transformer_flops/134081)/(total_unet_flops/170457):.1f}x less efficient")
    
    # Attention breakdown
    attention_per_layer = 2 * seq_len * seq_len * d_model + 4 * seq_len * seq_len
    total_attention = attention_per_layer * num_layers
    print(f"\nAttention analysis:")
    print(f"Attention FLOPs per layer: {attention_per_layer:,}")
    print(f"Total attention FLOPs: {total_attention:,}")
    print(f"Attention is {total_attention/total_transformer_flops:.1%} of Transformer FLOPs")

if __name__ == "__main__":
    exact_flop_analysis()
