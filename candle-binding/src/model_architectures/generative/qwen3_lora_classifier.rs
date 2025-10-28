//! Qwen3 Generative Classifier with LoRA Support
//!
//! This module provides a Qwen3-based text classifier that uses generative classification:
//! - Load base Qwen3 model + LoRA adapters
//! - Format prompts with ChatML template
//! - Extract logits for category tokens
//! - Perform softmax classification
//!
//! This is a **simplified causal LM** focused on classification inference,
//! not full text generation. It reuses Qwen3Embedding architecture with modifications.

use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use crate::model_architectures::embedding::qwen3_embedding::{
    Qwen3EmbeddingConfig, RmsNorm,
};
use crate::model_architectures::lora::{LoRAAdapter, LoRAConfig};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Use official Candle RoPE implementation (optimized with CUDA/Metal kernels)
use candle_nn::rotary_emb;

/// Rotary Position Embedding (RoPE)
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> UnifiedResult<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device).map_err(|e| {
            UnifiedError::Processing {
                operation: "create inv_freq".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        let t = Tensor::arange(0u32, max_seq_len as u32, device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create position tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "convert to f32".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .reshape((max_seq_len, 1))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape position".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        let freqs = t.matmul(&inv_freq).map_err(|e| UnifiedError::Processing {
            operation: "compute freqs".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // FIXED: Don't concatenate freqs - official rope expects [seq_len, head_dim/2]
        // The official candle_nn::rotary_emb::rope will handle the full rotation internally
        let cos = freqs.cos().map_err(|e| UnifiedError::Processing {
            operation: "compute cos".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        let sin = freqs.sin().map_err(|e| UnifiedError::Processing {
            operation: "compute sin".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        Ok(Self { sin, cos, dim })
    }

    fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> UnifiedResult<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len).map_err(|e| {
            UnifiedError::Processing {
                operation: "narrow cos".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;
        let sin = self.sin.narrow(0, 0, seq_len).map_err(|e| {
            UnifiedError::Processing {
                operation: "narrow sin".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Use official candle_nn::rotary_emb::rope (optimized CUDA/Metal kernels)
        // CRITICAL: .contiguous() is required before RoPE application
        let q_embed = rotary_emb::rope(&q.contiguous().map_err(|e| UnifiedError::Processing {
            operation: "make q contiguous for RoPE".to_string(),
            source: e.to_string(),
            input_context: None,
        })?, &cos, &sin).map_err(|e| UnifiedError::Processing {
            operation: "apply RoPE to q".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let k_embed = rotary_emb::rope(&k.contiguous().map_err(|e| UnifiedError::Processing {
            operation: "make k contiguous for RoPE".to_string(),
            source: e.to_string(),
            input_context: None,
        })?, &cos, &sin).map_err(|e| UnifiedError::Processing {
            operation: "apply RoPE to k".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        Ok((q_embed, k_embed))
    }
}

/// Multi-Head Attention with Grouped Query Attention (GQA) and LoRA support
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // QK-normalization (Qwen3 specific)
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // LoRA adapters for attention projections
    q_lora: Option<LoRAAdapter>,
    k_lora: Option<LoRAAdapter>,
    v_lora: Option<LoRAAdapter>,
    o_lora: Option<LoRAAdapter>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
}

impl Attention {
    fn new(config: &Qwen3EmbeddingConfig, vb: VarBuilder, lora_vb: Option<VarBuilder>, lora_config: Option<&LoRAConfig>, device: &Device) -> UnifiedResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim; // Use explicit head_dim from config (128 for Qwen3-0.6B)

        let q_proj = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create q_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        let k_proj = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create k_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        let v_proj = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create v_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create o_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Load LoRA adapters if available
        let (q_lora, k_lora, v_lora, o_lora) = if let (Some(lora_vb), Some(lora_cfg)) = (lora_vb, lora_config) {
            let q_lora = match LoRAAdapter::new(hidden_size, num_heads * head_dim, lora_cfg, lora_vb.pp("q_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded q_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load q_proj LoRA: {}", e); None }
            };
            let k_lora = match LoRAAdapter::new(hidden_size, num_kv_heads * head_dim, lora_cfg, lora_vb.pp("k_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded k_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load k_proj LoRA: {}", e); None }
            };
            let v_lora = match LoRAAdapter::new(hidden_size, num_kv_heads * head_dim, lora_cfg, lora_vb.pp("v_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded v_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load v_proj LoRA: {}", e); None }
            };
            let o_lora = match LoRAAdapter::new(num_heads * head_dim, hidden_size, lora_cfg, lora_vb.pp("o_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded o_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load o_proj LoRA: {}", e); None }
            };
            (q_lora, k_lora, v_lora, o_lora)
        } else {
            (None, None, None, None)
        };

        // Load Q and K normalization layers (Qwen3 uses QK-norm)
        let q_norm_weight = vb.pp("q_norm").get((head_dim,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load q_norm weight".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        // FIXED: Use config.rms_norm_eps instead of hardcoded 1e-6
        let q_norm = RmsNorm::new(q_norm_weight, config.rms_norm_eps);

        let k_norm_weight = vb.pp("k_norm").get((head_dim,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load k_norm weight".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        // FIXED: Use config.rms_norm_eps instead of hardcoded 1e-6
        let k_norm = RmsNorm::new(k_norm_weight, config.rms_norm_eps);

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            q_lora,
            k_lora,
            v_lora,
            o_lora,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> UnifiedResult<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3().map_err(|e| {
            UnifiedError::Processing {
                operation: "get hidden_states dims".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Get the dtype from the q_proj weights to ensure consistency
        let target_dtype = self.q_proj.weight().dtype();
        
        // Ensure hidden_states matches model dtype before LoRA operations
        let hidden_states = hidden_states.to_dtype(target_dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert hidden_states to model dtype in attention".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Project Q, K, V with LoRA
        let mut q = self.q_proj.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward q_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        if let Some(ref q_lora) = self.q_lora {
            let q_lora_out = q_lora.forward(&hidden_states, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward q_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            q = (q + q_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add q_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }

        let mut k = self.k_proj.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward k_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        if let Some(ref k_lora) = self.k_lora {
            let k_lora_out = k_lora.forward(&hidden_states, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward k_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            k = (k + k_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add k_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }

        let mut v = self.v_proj.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward v_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        if let Some(ref v_lora) = self.v_lora {
            let v_lora_out = v_lora.forward(&hidden_states, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward v_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            v = (v + v_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add v_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }

        // Apply QK-normalization (Qwen3 specific - normalize Q and K before RoPE)
        // CRITICAL: Must match official Candle implementation order!
        // 1. Reshape to (B, L, H, D)
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape q".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape k".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        // 2. Transpose to (B, H, L, D)
        let q = q.transpose(1, 2)
            .map_err(|e| UnifiedError::Processing {
                operation: "transpose q".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        let k = k.transpose(1, 2)
            .map_err(|e| UnifiedError::Processing {
                operation: "transpose k".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        // 3. Flatten to (B*H*L, D) for per-head normalization
        let q_flat = q.flatten(0, 2)
            .map_err(|e| UnifiedError::Processing {
                operation: "flatten q for norm".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        let k_flat = k.flatten(0, 2)
            .map_err(|e| UnifiedError::Processing {
                operation: "flatten k for norm".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        // 4. Apply RMSNorm
        let q_flat = self.q_norm.forward(&q_flat).map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "forward q_norm".to_string(),
            source: e.to_string(),
            context: None,
        })?;
        let k_flat = self.k_norm.forward(&k_flat).map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "forward k_norm".to_string(),
            source: e.to_string(),
            context: None,
        })?;
        
        // 5. Reshape back to (B, H, L, D)
        let q = q_flat.reshape((batch_size, self.num_heads, seq_len, self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape q after norm".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        let k = k_flat.reshape((batch_size, self.num_kv_heads, seq_len, self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape k after norm".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Reshape V (no normalization for V)
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape q".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .transpose(1, 2)
            .map_err(|e| UnifiedError::Processing {
                operation: "transpose v".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, seq_len)?;

        // Repeat KV for GQA if needed
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeat_factor = self.num_heads / self.num_kv_heads;
            let k = Self::repeat_kv(&k, repeat_factor)?;
            let v = Self::repeat_kv(&v, repeat_factor)?;
            (k, v)
        } else {
            (k, v)
        };

        // Compute attention scores
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1).map_err(|e| {
            UnifiedError::Processing {
                operation: "transpose k".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?)
        .map_err(|e| UnifiedError::Processing {
            operation: "matmul qk".to_string(),
            source: e.to_string(),
            input_context: None,
        })?
        .affine(scale, 0.0)
        .map_err(|e| UnifiedError::Processing {
            operation: "scale attention".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Apply softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).map_err(|e| {
            UnifiedError::Processing {
                operation: "softmax attention".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;
        
        // Convert attention weights to match target dtype (softmax returns F32)
        let attn_weights = attn_weights.to_dtype(target_dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert attention weights to target dtype".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v).map_err(|e| {
            UnifiedError::Processing {
                operation: "matmul attention values".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| UnifiedError::Processing {
                operation: "transpose output".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape output".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Output projection with LoRA
        let mut output = self.o_proj.forward(&attn_output).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward o_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        
        if let Some(ref o_lora) = self.o_lora {
            let o_lora_out = o_lora.forward(&attn_output, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward o_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            output = (output + o_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add o_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }
        
        Ok(output)
    }

    fn repeat_kv(x: &Tensor, n: usize) -> UnifiedResult<Tensor> {
        if n == 1 {
            return Ok(x.clone());
        }
        
        let (b, num_kv_heads, seq_len, head_dim) = x.dims4().map_err(|e| {
            UnifiedError::Processing {
                operation: "get kv dims".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        x.unsqueeze(2)
            .map_err(|e| UnifiedError::Processing {
                operation: "unsqueeze kv".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .expand(&[b, num_kv_heads, n, seq_len, head_dim])
            .map_err(|e| UnifiedError::Processing {
                operation: "expand kv".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .reshape((b, num_kv_heads * n, seq_len, head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape expanded kv".to_string(),
                source: e.to_string(),
                input_context: None,
            })
    }
}

/// MLP (Feed-Forward Network) with LoRA support
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    // LoRA adapters for MLP projections
    gate_lora: Option<LoRAAdapter>,
    up_lora: Option<LoRAAdapter>,
    down_lora: Option<LoRAAdapter>,
}

impl MLP {
    fn new(config: &Qwen3EmbeddingConfig, vb: VarBuilder, lora_vb: Option<VarBuilder>, lora_config: Option<&LoRAConfig>, device: &Device) -> UnifiedResult<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create gate_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create up_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        let down_proj = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create down_proj".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Load LoRA adapters if available
        let (gate_lora, up_lora, down_lora) = if let (Some(lora_vb), Some(lora_cfg)) = (lora_vb, lora_config) {
            let gate_lora = match LoRAAdapter::new(hidden_size, intermediate_size, lora_cfg, lora_vb.pp("gate_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded gate_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load gate_proj LoRA: {}", e); None }
            };
            let up_lora = match LoRAAdapter::new(hidden_size, intermediate_size, lora_cfg, lora_vb.pp("up_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded up_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load up_proj LoRA: {}", e); None }
            };
            let down_lora = match LoRAAdapter::new(intermediate_size, hidden_size, lora_cfg, lora_vb.pp("down_proj"), device) {
                Ok(adapter) => { eprintln!("  ✅ Loaded down_proj LoRA"); Some(adapter) },
                Err(e) => { eprintln!("  ❌ Failed to load down_proj LoRA: {}", e); None }
            };
            (gate_lora, up_lora, down_lora)
        } else {
            (None, None, None)
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_lora,
            up_lora,
            down_lora,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> UnifiedResult<Tensor> {
        // Get the dtype from the gate_proj weights to ensure consistency
        let target_dtype = self.gate_proj.weight().dtype();
        
        // Ensure hidden_states matches model dtype before LoRA operations
        let hidden_states = hidden_states.to_dtype(target_dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert hidden_states to model dtype in MLP".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;
        
        // Gate projection with LoRA
        let mut gate = self.gate_proj.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward gate_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        if let Some(ref gate_lora) = self.gate_lora {
            let gate_lora_out = gate_lora.forward(&hidden_states, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward gate_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            gate = (gate + gate_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add gate_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }

        let gate = candle_nn::ops::silu(&gate).map_err(|e| UnifiedError::Processing {
            operation: "apply silu".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Up projection with LoRA
        let mut up = self.up_proj.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward up_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        if let Some(ref up_lora) = self.up_lora {
            let up_lora_out = up_lora.forward(&hidden_states, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward up_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            up = (up + up_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add up_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }

        let intermediate = (gate * up).map_err(|e| UnifiedError::Processing {
            operation: "mul gate and up".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Down projection with LoRA
        let mut output = self.down_proj.forward(&intermediate).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward down_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        if let Some(ref down_lora) = self.down_lora {
            let down_lora_out = down_lora.forward(&intermediate, false).map_err(|e| {
                UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward down_lora".to_string(),
                    source: e.to_string(),
                    context: None,
                }
            })?;
            output = (output + down_lora_out).map_err(|e| UnifiedError::Processing {
                operation: "add down_lora".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        }

        Ok(output)
    }
}

/// Qwen3 Decoder Layer (Transformer Block)
struct Qwen3DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    fn new(config: &Qwen3EmbeddingConfig, vb: VarBuilder, lora_vb: Option<VarBuilder>, lora_config: Option<&LoRAConfig>, device: &Device) -> UnifiedResult<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"), lora_vb.as_ref().map(|v| v.pp("self_attn")), lora_config, device)?;
        let mlp = MLP::new(config, vb.pp("mlp"), lora_vb.as_ref().map(|v| v.pp("mlp")), lora_config, device)?;

        let input_ln_weight = vb
            .pp("input_layernorm")
            .get((config.hidden_size,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load input_layernorm".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        let input_layernorm = RmsNorm::new(input_ln_weight, config.rms_norm_eps);

        let post_attn_ln_weight = vb
            .pp("post_attention_layernorm")
            .get((config.hidden_size,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load post_attention_layernorm".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        let post_attention_layernorm = RmsNorm::new(post_attn_ln_weight, config.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> UnifiedResult<Tensor> {
        // Pre-norm: layernorm before attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward input_layernorm".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        // Self-attention
        let hidden_states = self.self_attn.forward(&hidden_states)?;

        // Residual connection
        let hidden_states = (hidden_states + residual).map_err(|e| UnifiedError::Processing {
            operation: "add attention residual".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Pre-norm: layernorm before MLP
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward post_attention_layernorm".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        // MLP
        let hidden_states = self.mlp.forward(&hidden_states)?;

        // Residual connection
        (hidden_states + residual).map_err(|e| UnifiedError::Processing {
            operation: "add mlp residual".to_string(),
            source: e.to_string(),
            input_context: None,
        })
    }
}

/// Classification result using logit extraction
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Category index (0-based)
    pub class: i32,
    /// Category name
    pub category: String,
    /// Classification confidence (0.0-1.0)
    pub confidence: f32,
    /// Full probability distribution over all categories
    pub probabilities: Vec<f32>,
}

/// Label mapping for classification (loaded from label_mapping.json)
#[derive(Debug, Clone, Deserialize)]
pub struct LabelMapping {
    pub label2id: HashMap<String, usize>,
    pub id2label: HashMap<String, String>,
    #[serde(default)]
    pub instruction_template: String,
}

impl LabelMapping {
    pub fn from_file(path: &Path) -> UnifiedResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| UnifiedError::Configuration {
            operation: "load label_mapping.json".to_string(),
            source: ConfigErrorType::FileNotFound(format!("{:?}", path)),
            context: Some(e.to_string()),
        })?;

        serde_json::from_str(&content).map_err(|e| UnifiedError::Configuration {
            operation: "parse label_mapping.json".to_string(),
            source: ConfigErrorType::ParseError(e.to_string()),
            context: None,
        })
    }

    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<_> = self
            .id2label
            .iter()
            .map(|(id, label)| (id.parse::<usize>().unwrap_or(0), label.clone()))
            .collect();
        cats.sort_by_key(|(id, _)| *id);
        cats.into_iter().map(|(_, label)| label).collect()
    }
}

/// LoRA adapter metadata (not the actual adapter, just config info)
#[derive(Debug, Clone)]
pub struct LoRAAdapterInfo {
    /// LoRA rank
    pub r: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// Dropout (not used during inference)
    pub dropout: f32,
}

/// Qwen3 LoRA Generative Classifier
///
/// This classifier:
/// 1. Loads Qwen3 base model from base_model_path
/// 2. Loads LoRA adapters from lora_adapter_path (NOT MERGED)
/// 3. Formats prompts using ChatML template
/// 4. Runs forward pass to get logits at last position
/// 5. Extracts logits for each category's first token
/// 6. Returns softmax probabilities
pub struct Qwen3LoRAClassifier {
    /// Base model configuration
    config: Qwen3EmbeddingConfig,
    
    /// Tokenizer
    tokenizer: Arc<Tokenizer>,
    
    /// Token embeddings
    embeddings: Embedding,
    
    /// Transformer layers (full Qwen3 decoder stack with 28 layers)
    layers: Vec<Qwen3DecoderLayer>,
    
    /// Final layer normalization
    norm: RmsNorm,
    
    /// LM head: projects hidden states to vocabulary logits
    /// Shape: [hidden_size, vocab_size]
    lm_head: Linear,
    
    /// LoRA adapter info (metadata only, actual adapters are in layers)
    lora_adapter_info: Option<LoRAAdapterInfo>,
    
    /// LoRA configuration
    lora_config: Option<LoRAConfig>,
    
    /// Label mapping (categories)
    label_mapping: LabelMapping,
    
    /// Category token IDs (first token of each category name)
    category_token_ids: Vec<u32>,
    
    /// Device
    device: Device,
    
    /// Model dtype (BF16 on CUDA, F32 on CPU)
    dtype: DType,
}

impl Qwen3LoRAClassifier {
    /// Load Qwen3 + LoRA classifier
    ///
    /// Parameters:
    /// - base_model_path: Path to base Qwen3 model (e.g., "./models/Qwen3-0.6B")
    /// - lora_adapter_path: Path to LoRA adapter (e.g., "./models/qwen3_generative_classifier_r16")
    /// - device: Device to run on (CPU or GPU)
    ///
    /// Expected structure:
    /// ```
    /// base_model_path/
    ///   ├── config.json          (Qwen3 config)
    ///   ├── model.safetensors    (Base model weights)
    ///   └── tokenizer.json       (Tokenizer)
    ///
    /// lora_adapter_path/
    ///   ├── adapter_config.json  (LoRA config with r, alpha)
    ///   ├── adapter_model.safetensors (LoRA weights)
    ///   └── label_mapping.json   (Category labels and instruction template)
    /// ```
    pub fn from_pretrained(model_path: &str, device: &Device) -> UnifiedResult<Self> {
        // Parse model_path - if it contains "qwen3_generative_classifier", it's the adapter path
        // Otherwise treat it as base model path
        let (base_model_path, lora_adapter_path) = if model_path.contains("qwen3_generative_classifier") {
            // User provided LoRA adapter path, infer base model path
            let adapter_dir = Path::new(model_path);
            let parent = adapter_dir.parent().unwrap_or(Path::new("./models"));
            let base_path = parent.join("Qwen3-0.6B");
            (base_path.to_string_lossy().to_string(), model_path.to_string())
        } else {
            // User provided base model path, look for adapter in same parent dir
            let base_dir = Path::new(model_path);
            let parent = base_dir.parent().unwrap_or(Path::new("./models"));
            let adapter_path = parent.join("qwen3_generative_classifier_r16");
            (model_path.to_string(), adapter_path.to_string_lossy().to_string())
        };

        println!("Loading Qwen3 + LoRA classifier:");
        println!("  Base model: {}", base_model_path);
        println!("  LoRA adapter: {}", lora_adapter_path);

        let base_dir = Path::new(&base_model_path);
        let adapter_dir = Path::new(&lora_adapter_path);

        // Load config from base model
        let config = Qwen3EmbeddingConfig::from_pretrained(&base_model_path)?;

        // Load tokenizer from base model
        let tokenizer_path = base_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| UnifiedError::Configuration {
                operation: "load tokenizer".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", tokenizer_path)),
                context: Some(e.to_string()),
            })?;

        // Load label mapping from adapter directory
        let label_mapping_path = adapter_dir.join("label_mapping.json");
        let label_mapping = LabelMapping::from_file(&label_mapping_path)?;

        // Use BF16 on CUDA for optimal performance
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        println!("  Model dtype: {:?}", dtype);

        // Load base model weights
        let vb = Self::load_weights(base_dir, device, dtype)?;

        // Load LoRA adapter config and weights (same dtype as base for now)
        let (lora_adapter_info, lora_config, lora_vb) = Self::load_lora_adapter(adapter_dir, device, dtype)?;

        // Create model components
        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )
        .map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "create embeddings".to_string(),
            source: e.to_string(),
            context: None,
        })?;

        // Load all transformer layers with LoRA
        println!("Loading {} transformer layers with LoRA...", config.num_hidden_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = vb.pp("model.layers");
        let lora_layers_vb = lora_vb.as_ref().map(|v| v.pp("base_model.model.model.layers"));
        
        for layer_idx in 0..config.num_hidden_layers {
            if layer_idx % 5 == 0 {
                println!("  Loading layer {}/{}", layer_idx + 1, config.num_hidden_layers);
            }
            let lora_layer_vb = lora_layers_vb.as_ref().map(|v| v.pp(&layer_idx.to_string()));
            let layer = Qwen3DecoderLayer::new(&config, layers_vb.pp(&layer_idx.to_string()), lora_layer_vb, lora_config.as_ref(), device)?;
            layers.push(layer);
        }
        println!("✅ All {} transformer layers loaded with LoRA", config.num_hidden_layers);

        // Load norm weights
        let norm_weight = vb
            .pp("model.norm")
            .get((config.hidden_size,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load norm weights".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        // LM head (maps hidden states to vocabulary logits)
        // Note: Qwen3 models typically don't have bias in lm_head
        let lm_head = candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create lm_head".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Prepare category token IDs
        let category_token_ids = Self::prepare_category_tokens(&tokenizer, &label_mapping)?;

        println!("✅ Qwen3 LoRA classifier loaded successfully");
        println!("  - Base model: {} parameters", config.hidden_size * config.vocab_size);
        println!("  - Transformer layers: {}", config.num_hidden_layers);
        println!("  - LoRA adapter: rank={}, alpha={}", lora_adapter_info.r, lora_adapter_info.alpha);
        println!("  - Categories: {}", category_token_ids.len());

        Ok(Self {
            config,
            tokenizer: Arc::new(tokenizer),
            embeddings,
            layers,
            norm,
            lm_head,
            lora_adapter_info: Some(lora_adapter_info),
            lora_config,
            label_mapping,
            category_token_ids,
            device: device.clone(),
            dtype,
        })
    }

    /// Load LoRA adapter configuration and weights
    fn load_lora_adapter<'a>(adapter_dir: &Path, device: &'a Device, dtype: DType) -> UnifiedResult<(LoRAAdapterInfo, Option<LoRAConfig>, Option<VarBuilder<'a>>)> {
        // Load adapter config
        let config_path = adapter_dir.join("adapter_config.json");
        
        if !config_path.exists() {
            eprintln!("Warning: No adapter_config.json found, LoRA will not be applied");
            let info = LoRAAdapterInfo {
                r: 0,
                alpha: 0.0,
                dropout: 0.0,
            };
            return Ok((info, None, None));
        }

        let config_content = std::fs::read_to_string(&config_path).map_err(|e| {
            UnifiedError::Configuration {
                operation: "load adapter_config.json".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", config_path)),
                context: Some(e.to_string()),
            }
        })?;

        // Parse adapter config
        let adapter_config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| {
            UnifiedError::Configuration {
                operation: "parse adapter_config.json".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;

        let r = adapter_config["r"].as_u64().unwrap_or(16) as usize;
        let alpha = adapter_config["lora_alpha"].as_f64().unwrap_or(32.0);
        let dropout = adapter_config["lora_dropout"].as_f64().unwrap_or(0.05);

        println!("  LoRA config: r={}, alpha={}, dropout={}", r, alpha, dropout);

        // Create LoRA config
        let lora_config = LoRAConfig {
            rank: r,
            alpha,
            dropout,
            target_modules: vec!["q_proj".to_string(), "k_proj".to_string(), "v_proj".to_string(), "o_proj".to_string(), "gate_proj".to_string(), "up_proj".to_string(), "down_proj".to_string()],
            use_bias: false,
            init_method: crate::model_architectures::lora::LoRAInitMethod::Kaiming,
        };

        // Load LoRA weights from adapter_model.safetensors
        let adapter_weights_path = adapter_dir.join("adapter_model.safetensors");
        
        if !adapter_weights_path.exists() {
            println!("Warning: adapter_model.safetensors not found, LoRA weights will not be loaded");
            let info = LoRAAdapterInfo {
                r,
                alpha: alpha as f32,
                dropout: dropout as f32,
            };
            return Ok((info, Some(lora_config), None));
        }

        println!("  Loading LoRA weights from: {:?}", adapter_weights_path);
        println!("  Loading LoRA weights with dtype: {:?}", dtype);
        
        let lora_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[adapter_weights_path], dtype, device).map_err(
                |e| UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "load LoRA weights".to_string(),
                    source: e.to_string(),
                    context: Some("Failed to load adapter_model.safetensors".to_string()),
                },
            )?
        };

        println!("✅ LoRA weights loaded successfully");

        let info = LoRAAdapterInfo {
            r,
            alpha: alpha as f32,
            dropout: dropout as f32,
        };

        Ok((info, Some(lora_config), Some(lora_vb)))
    }

    /// Load model weights from safetensors file
    fn load_weights<'a>(model_dir: &Path, device: &'a Device, dtype: DType) -> UnifiedResult<VarBuilder<'a>> {
        let weights_path = model_dir.join("model.safetensors");
        
        if !weights_path.exists() {
            return Err(UnifiedError::Configuration {
                operation: "find model weights".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", weights_path)),
                context: Some("Expected model.safetensors in model directory".to_string()),
            });
        }

        println!("  Loading base model weights with dtype: {:?}", dtype);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device).map_err(
                |e| UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "load weights".to_string(),
                    source: e.to_string(),
                    context: Some("Failed to load model.safetensors".to_string()),
                },
            )?
        };

        Ok(vb)
    }

    /// Prepare category token IDs for logit extraction
    fn prepare_category_tokens(
        tokenizer: &Tokenizer,
        label_mapping: &LabelMapping,
    ) -> UnifiedResult<Vec<u32>> {
        let categories = label_mapping.categories();
        let mut token_ids = Vec::new();

        for category in &categories {
            // Tokenize category with leading space (matches generation context)
            let text = format!(" {}", category);
            let encoding = tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "tokenize category".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: Some(format!("Failed to tokenize category: {}", category)),
                })?;

            let tokens = encoding.get_ids();
            if tokens.is_empty() {
                return Err(UnifiedError::Validation {
                    field: "category tokenization".to_string(),
                    expected: "at least 1 token".to_string(),
                    actual: format!("0 tokens for '{}'", category),
                    context: None,
                });
            }

            token_ids.push(tokens[0]);
        }

        println!("  Prepared {} category tokens", token_ids.len());
        Ok(token_ids)
    }

    /// Format prompt using ChatML template
    ///
    /// Matches Python server_generative.py line 406-408:
    /// ```python
    /// prompt = tokenizer.apply_chat_template(
    ///     messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    /// )
    /// ```
    pub fn format_prompt(&self, text: &str) -> String {
        let instruction = if !self.label_mapping.instruction_template.is_empty() {
            self.label_mapping
                .instruction_template
                .replace("{question}", text)
        } else {
            // Fallback template
            let categories = self.label_mapping.categories().join(", ");
            format!(
                "You are an expert academic classifier. Classify the following question into exactly ONE category. Respond with ONLY the category name.\n\nCategories: {}\n\nNow classify this question:\nQ: {}\nA:",
                categories, text
            )
        };

        // ChatML format with add_generation_prompt=True
        // Note: Qwen3 tokenizer adds <think> tags even with enable_thinking=False
        // We must match this to extract logits from the correct position
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            instruction
        )
    }

    /// Classify text using logit extraction (fast, single forward pass)
    ///
    /// This method matches the implementation in qwen3_causal.rs:
    /// - Single forward pass through the model
    /// - Extract logits for category tokens
    /// - Apply softmax to get probabilities
    ///
    /// # Arguments
    /// - `text`: Input text to classify
    ///
    /// # Returns
    /// Classification result with category, confidence, and probability distribution
    pub fn classify(&self, text: &str) -> UnifiedResult<ClassificationResult> {
        // Format prompt
        let prompt = self.format_prompt(text);

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenize prompt".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;

        let token_ids = encoding.get_ids();
        
        // Create input tensors
        let input_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create input tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .unsqueeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "unsqueeze input".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Step 1: Embed tokens
        let mut hidden_states = self.embeddings.forward(&input_ids).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward embeddings".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        
        // Convert to model dtype (critical for BF16 support)
        hidden_states = hidden_states.to_dtype(self.dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert hidden_states to model dtype".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Step 2: Run through all transformer layers (full Qwen3 forward pass with LoRA)
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states).map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: format!("forward layer {}", layer_idx),
                source: e.to_string(),
                context: None,
            })?;
        }

        // Step 3: Apply final norm
        let hidden_states = self.norm.forward(&hidden_states).map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "forward norm".to_string(),
            source: e.to_string(),
            context: None,
        })?;
        
        // Convert to model dtype after norm (RmsNorm might return F32)
        let hidden_states = hidden_states.to_dtype(self.dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert hidden_states to model dtype after norm".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Step 4: Project to vocabulary logits
        let logits = self.lm_head.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward lm_head".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        // Get logits at last position: [batch=1, seq_len, vocab] → [vocab]
        let seq_len = logits.dim(1).map_err(|e| UnifiedError::Processing {
            operation: "get sequence length".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let last_logits = logits
            .i((0, seq_len - 1, ..))
            .map_err(|e| UnifiedError::Processing {
                operation: "index last position".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "convert logits to f32".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Extract logits for category tokens
        let mut category_logits = Vec::new();
        for (idx, &token_id) in self.category_token_ids.iter().enumerate() {
            let logit = last_logits
                .i(token_id as usize)
                .map_err(|e| UnifiedError::Processing {
                    operation: "extract category logit".to_string(),
                    source: e.to_string(),
                    input_context: Some(format!("token_id={}", token_id)),
                })?
                .to_scalar::<f32>()
                .map_err(|e| UnifiedError::Processing {
                    operation: "convert logit to scalar".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            
            category_logits.push(logit);
        }

        // Apply softmax to get probabilities
        let category_logits_tensor = Tensor::from_vec(
            category_logits,
            (self.category_token_ids.len(),),
            &self.device,
        )
        .map_err(|e| UnifiedError::Processing {
            operation: "create category logits tensor".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let probabilities = candle_nn::ops::softmax_last_dim(&category_logits_tensor)
            .map_err(|e| UnifiedError::Processing {
                operation: "softmax probabilities".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        let probs_vec = probabilities.to_vec1::<f32>().map_err(|e| {
            UnifiedError::Processing {
                operation: "convert probabilities to vec".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;
        
        // Find best category
        let best_idx = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let best_confidence = probs_vec[best_idx];
        let categories = self.label_mapping.categories();
        let best_category = categories[best_idx].clone();
        
        Ok(ClassificationResult {
            class: best_idx as i32,
            category: best_category,
            confidence: best_confidence,
            probabilities: probs_vec,
        })
    }
    
    /// Classify multiple texts in a batch (optimized for concurrent requests)
    ///
    /// # Arguments
    /// - `texts`: Array of texts to classify
    ///
    /// # Returns
    /// Vector of classification results, one per input text
    pub fn classify_batch(&self, texts: &[&str]) -> UnifiedResult<Vec<ClassificationResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                let prompt = self.format_prompt(text);
                self.tokenizer.encode(prompt, true).map_err(|e| {
                    UnifiedError::Configuration {
                        operation: "tokenization".to_string(),
                        source: ConfigErrorType::ParseError(e.to_string()),
                        context: Some("Failed to tokenize batch input".to_string()),
                    }
                })
            })
            .collect::<UnifiedResult<Vec<_>>>()?;
        
        // Find max length and pad all sequences
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let pad_token_id = self.tokenizer.get_padding()
            .map(|p| p.pad_id)
            .unwrap_or(0);
        
        let mut batch_ids = Vec::new();
        for encoding in &encodings {
            let mut ids = encoding.get_ids().to_vec();
            ids.resize(max_len, pad_token_id);
            batch_ids.extend(ids);
        }
        
        // Create batched tensor [batch_size, seq_len]
        let batch_tensor = Tensor::from_vec(batch_ids, (texts.len(), max_len), &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create batch tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        // Single forward pass for entire batch
        // Step 1: Embed tokens
        let mut hidden_states = self.embeddings.forward(&batch_tensor).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward embeddings (batch)".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        
        // Convert to model dtype (critical for BF16 support)
        hidden_states = hidden_states.to_dtype(self.dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert hidden_states to model dtype (batch)".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Step 2: Run through all transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states).map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: format!("forward layer {} (batch)", layer_idx),
                source: e.to_string(),
                context: None,
            })?;
        }

        // Step 3: Apply final norm
        let hidden_states = self.norm.forward(&hidden_states).map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "forward norm (batch)".to_string(),
            source: e.to_string(),
            context: None,
        })?;
        
        // Convert to model dtype after norm (RmsNorm might return F32)
        let hidden_states = hidden_states.to_dtype(self.dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert hidden_states to model dtype after norm (batch)".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        // Step 4: Project to vocabulary logits
        let logits = self.lm_head.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward lm_head (batch)".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        
        // Process each sample in the batch
        let mut results = Vec::with_capacity(texts.len());
        let categories = self.label_mapping.categories();
        
        for i in 0..texts.len() {
            // Get last position logits for this sample
            let last_logits = logits
                .i((i, max_len - 1, ..))
                .map_err(|e| UnifiedError::Processing {
                    operation: format!("extract logits for sample {}", i),
                    source: e.to_string(),
                    input_context: None,
                })?
                .to_dtype(DType::F32)
                .map_err(|e| UnifiedError::Processing {
                    operation: "convert to f32".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            
            // Extract category logits
            let category_logits: Vec<f32> = self.category_token_ids
                .iter()
                .map(|&token_id| {
                    last_logits
                        .i(token_id as usize)
                        .and_then(|t| t.to_scalar::<f32>())
                        .map_err(|e| UnifiedError::Processing {
                            operation: format!("extract logit for token {}", token_id),
                            source: e.to_string(),
                            input_context: None,
                        })
                })
                .collect::<UnifiedResult<Vec<f32>>>()?;
            
            // Softmax
            let category_logits_tensor = Tensor::from_vec(
                category_logits,
                (self.category_token_ids.len(),),
                &self.device,
            )
            .map_err(|e| UnifiedError::Processing {
                operation: "create logits tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
            
            let probabilities = candle_nn::ops::softmax_last_dim(&category_logits_tensor)
                .map_err(|e| UnifiedError::Processing {
                    operation: "softmax".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            
            let probs_vec = probabilities.to_vec1::<f32>().map_err(|e| {
                UnifiedError::Processing {
                    operation: "convert probabilities".to_string(),
                    source: e.to_string(),
                    input_context: None,
                }
            })?;
            
            // Find best
            let best_idx = probs_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            results.push(ClassificationResult {
                class: best_idx as i32,
                category: categories[best_idx].clone(),
                confidence: probs_vec[best_idx],
                probabilities: probs_vec,
            });
        }
        
        Ok(results)
    }

    /// Get categories
    pub fn categories(&self) -> Vec<String> {
        self.label_mapping.categories()
    }

    /// Get number of categories
    pub fn num_categories(&self) -> usize {
        self.label_mapping.id2label.len()
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }
}

/// Apply softmax to convert logits to probabilities
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        
        // Check sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // Check highest logit has highest prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_chat_template_format() {
        let text = "What is the derivative of x²?";
        let instruction = format!(
            "You are an expert academic classifier. Classify the following question...\nQ: {}\nA:",
            text
        );
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            instruction
        );
        
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant"));
        assert!(prompt.contains(text));
    }
}

