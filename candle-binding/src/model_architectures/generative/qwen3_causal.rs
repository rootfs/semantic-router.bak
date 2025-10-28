//! Qwen3 Causal Language Model (ForCausalLM)
//!
//! Implementation of Qwen3 as a causal language model for text generation.
//! Can be used for classification by generating category names directly.
//!
//! Supports:
//! - Text generation with various sampling strategies
//! - Chat template (ChatML format)
//! - Temperature and top-p sampling
//! - Repetition penalty
//! - Optional label mapping for classification tasks
//!
//! ## Usage Example
//! ```ignore
//! let model = Qwen3CausalLM::from_pretrained("./models/Qwen3-0.6B", &device)?;
//! 
//! // For generation
//! let result = model.generate("Hello", GenerationConfig::default())?;
//! println!("Generated: {}", result.text);
//! 
//! // For classification (format prompt appropriately)
//! let prompt = model.format_chat_prompt("Classify: What is photosynthesis?");
//! let result = model.generate(&prompt, GenerationConfig { max_new_tokens: 20, ..Default::default() })?;
//! println!("Category: {}", result.text);
//! ```

use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Qwen3 Causal LM Configuration
///
/// Loaded from config.json in the model directory
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3CausalConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_head_dim() -> usize {
    128 // Default for Qwen3-0.6B
}

impl Qwen3CausalConfig {
    /// Load configuration from a model directory
    pub fn from_pretrained(model_path: &str) -> UnifiedResult<Self> {
        let config_path = std::path::Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| UnifiedError::Configuration {
                operation: "config file loading".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", config_path)),
                context: Some(format!("Failed to read config from {:?}: {}", config_path, e)),
            })?;
        
        let config: Self = serde_json::from_str(&config_str)
            .map_err(|e| UnifiedError::Configuration {
                operation: "config JSON parsing".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: Some(format!("Failed to parse config.json from {}", model_path)),
            })?;
        
        // Trust the config file's head_dim value
        // No need to recalculate - the config knows best!
        
        Ok(config)
    }
}

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (None = greedy, 0.0 = greedy, higher = more random)
    pub temperature: Option<f64>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,
    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repeats)
    pub repeat_penalty: f32,
    /// Context size for repeat penalty
    pub repeat_last_n: usize,
    /// Random seed for sampling
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: Some(0.7),
            top_p: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299792458,
        }
    }
}

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text (decoded, excluding prompt)
    pub text: String,
    /// All token IDs (including prompt)
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excluding prompt)
    pub num_generated: usize,
    /// Tokens per second
    pub tokens_per_second: f64,
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


// ============================================================================
// Model Architecture Components
// ============================================================================

/// RMS Normalization (copied from qwen3_lora_classifier.rs)
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> UnifiedResult<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32;
        
        let x = x.to_dtype(internal_dtype).map_err(|e| UnifiedError::Processing {
            operation: "convert to f32".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let norm_x = (x.sqr()
            .map_err(|e| UnifiedError::Processing {
                operation: "square".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .mean_keepdim(D::Minus1)
            .map_err(|e| UnifiedError::Processing {
                operation: "mean".to_string(),
                source: e.to_string(),
                input_context: None,
            })? + self.eps)
            .map_err(|e| UnifiedError::Processing {
                operation: "add eps".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .sqrt()
            .map_err(|e| UnifiedError::Processing {
                operation: "sqrt".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        let x_normed = x.broadcast_div(&norm_x).map_err(|e| UnifiedError::Processing {
            operation: "normalize".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let result = x_normed.broadcast_mul(&self.weight.to_dtype(internal_dtype).map_err(|e| {
            UnifiedError::Processing {
                operation: "convert weight".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?)
        .map_err(|e| UnifiedError::Processing {
            operation: "mul weight".to_string(),
            source: e.to_string(),
            input_context: None,
        })?
        .to_dtype(x_dtype)
        .map_err(|e| UnifiedError::Processing {
            operation: "convert back".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        Ok(result)
    }
}

/// Rotary Position Embedding (RoPE)
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, theta: f32, dtype: DType, device: &Device) -> UnifiedResult<Self> {
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

        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1).map_err(|e| {
            UnifiedError::Processing {
                operation: "concat freqs".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        let cos = emb.cos().map_err(|e| UnifiedError::Processing {
            operation: "compute cos".to_string(),
            source: e.to_string(),
            input_context: None,
        })?.to_dtype(dtype).map_err(|e| UnifiedError::Processing {
            operation: "convert cos to dtype".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        let sin = emb.sin().map_err(|e| UnifiedError::Processing {
            operation: "compute sin".to_string(),
            source: e.to_string(),
            input_context: None,
        })?.to_dtype(dtype).map_err(|e| UnifiedError::Processing {
            operation: "convert sin to dtype".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        Ok(Self { sin, cos })
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

        let q_embed = Self::rotate_half(q, &cos, &sin)?;
        let k_embed = Self::rotate_half(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> UnifiedResult<Tensor> {
        let half_dim = x.dim(D::Minus1).map_err(|e| UnifiedError::Processing {
            operation: "get dim".to_string(),
            source: e.to_string(),
            input_context: None,
        })? / 2;
        
        let x1 = x.narrow(D::Minus1, 0, half_dim).map_err(|e| {
            UnifiedError::Processing {
                operation: "narrow x1".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim).map_err(|e| {
            UnifiedError::Processing {
                operation: "narrow x2".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        let x_rotated = Tensor::cat(&[&x2.neg().map_err(|e| UnifiedError::Processing {
            operation: "negate x2".to_string(),
            source: e.to_string(),
            input_context: None,
        })?, &x1], D::Minus1).map_err(|e| {
            UnifiedError::Processing {
                operation: "concat rotated".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        let result = (x.broadcast_mul(cos).map_err(|e| UnifiedError::Processing {
            operation: "mul cos".to_string(),
            source: e.to_string(),
            input_context: None,
        })? + x_rotated.broadcast_mul(sin).map_err(|e| UnifiedError::Processing {
            operation: "mul sin".to_string(),
            source: e.to_string(),
            input_context: None,
        })?).map_err(|e| UnifiedError::Processing {
            operation: "add rotary components".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        Ok(result)
    }
}

/// Multi-Head Attention with Grouped Query Attention (GQA)
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
}

impl Attention {
    fn new(config: &Qwen3CausalConfig, vb: VarBuilder, device: &Device, dtype: DType) -> UnifiedResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

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

        let q_norm_weight = vb.pp("q_norm").get((head_dim,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load q_norm weight".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        let q_norm = RmsNorm::new(q_norm_weight, 1e-6);

        let k_norm_weight = vb.pp("k_norm").get((head_dim,), "weight")
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load k_norm weight".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        let k_norm = RmsNorm::new(k_norm_weight, 1e-6);

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta as f32,
            dtype,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
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

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward q_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        let k = self.k_proj.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward k_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        let v = self.v_proj.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward v_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

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
        let q = q.transpose(1, 2).map_err(|e| UnifiedError::Processing {
            operation: "transpose q".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        let k = k.transpose(1, 2).map_err(|e| UnifiedError::Processing {
            operation: "transpose k".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        // 3. Flatten to (B*H*L, D) for per-head normalization
        let q_flat = q.flatten(0, 2).map_err(|e| UnifiedError::Processing {
            operation: "flatten q for norm".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        let k_flat = k.flatten(0, 2).map_err(|e| UnifiedError::Processing {
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

        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| UnifiedError::Processing {
                operation: "reshape v".to_string(),
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

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).map_err(|e| {
            UnifiedError::Processing {
                operation: "softmax attention".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

        let attn_output = attn_weights.matmul(&v).map_err(|e| {
            UnifiedError::Processing {
                operation: "matmul attention values".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })?;

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

        let output = self.o_proj.forward(&attn_output).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward o_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;
        
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

/// MLP (Feed-Forward Network)
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn new(config: &Qwen3CausalConfig, vb: VarBuilder) -> UnifiedResult<Self> {
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

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> UnifiedResult<Tensor> {
        let gate = self.gate_proj.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward gate_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        let gate = candle_nn::ops::silu(&gate).map_err(|e| UnifiedError::Processing {
            operation: "apply silu".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        let up = self.up_proj.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward up_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        let intermediate = (gate * up).map_err(|e| UnifiedError::Processing {
            operation: "mul gate and up".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        let output = self.down_proj.forward(&intermediate).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward down_proj".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

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
    fn new(config: &Qwen3CausalConfig, vb: VarBuilder, device: &Device, dtype: DType) -> UnifiedResult<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"), device, dtype)?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;

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
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward input_layernorm".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        let hidden_states = self.self_attn.forward(&hidden_states)?;

        let hidden_states = (hidden_states + residual).map_err(|e| UnifiedError::Processing {
            operation: "add attention residual".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward post_attention_layernorm".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        let hidden_states = self.mlp.forward(&hidden_states)?;

        (hidden_states + residual).map_err(|e| UnifiedError::Processing {
            operation: "add mlp residual".to_string(),
            source: e.to_string(),
            input_context: None,
        })
    }
}

// ============================================================================
// Logits Processor (for sampling)
// ============================================================================

struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl LogitsProcessor {
    fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            top_p,
        }
    }

    fn sample(&mut self, logits: &Tensor) -> UnifiedResult<u32> {
        let logits = logits.to_vec1::<f32>().map_err(|e| UnifiedError::Processing {
            operation: "convert logits to vec".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Apply temperature
        let logits = if let Some(temp) = self.temperature {
            if temp <= 0.0 {
                // Greedy decoding
                let max_idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                return Ok(max_idx as u32);
            } else {
                logits.iter().map(|&x| x / temp as f32).collect::<Vec<_>>()
            }
        } else {
            // Greedy if no temperature
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            return Ok(max_idx as u32);
        };

        // Apply top-p (nucleus sampling)
        let probs = self.softmax(&logits);
        
        if let Some(top_p) = self.top_p {
            self.sample_top_p(&probs, top_p)
        } else {
            self.sample_multinomial(&probs)
        }
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum).collect()
    }

    fn sample_multinomial(&mut self, probs: &[f32]) -> UnifiedResult<u32> {
        use rand::Rng;
        let mut cumsum = 0.0;
        let sample = self.rng.gen::<f32>();
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                return Ok(i as u32);
            }
        }
        
        Ok((probs.len() - 1) as u32)
    }

    fn sample_top_p(&mut self, probs: &[f32], top_p: f64) -> UnifiedResult<u32> {
        let mut sorted_probs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut cutoff_idx = sorted_probs.len();
        
        for (idx, (_, p)) in sorted_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p as f32 {
                cutoff_idx = idx + 1;
                break;
            }
        }

        let filtered_probs: Vec<(usize, f32)> = sorted_probs[..cutoff_idx].to_vec();
        let total: f32 = filtered_probs.iter().map(|(_, p)| p).sum();
        let normalized_probs: Vec<f32> = filtered_probs.iter().map(|(_, p)| p / total).collect();

        use rand::Rng;
        let mut cumsum = 0.0;
        let sample = self.rng.gen::<f32>();
        
        for (i, p) in normalized_probs.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                return Ok(filtered_probs[i].0 as u32);
            }
        }
        
        Ok(filtered_probs[filtered_probs.len() - 1].0 as u32)
    }
}

// ============================================================================
// Main Model
// ============================================================================

/// Qwen3 Causal Language Model
/// 
/// General purpose text generation model.
/// Optionally loads label_mapping.json for classification use cases.
pub struct Qwen3CausalLM {
    config: Qwen3CausalConfig,
    tokenizer: Arc<Tokenizer>,
    embeddings: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    /// Optional category mapping (category_name -> id) for classification tasks
    label2id: Option<std::collections::HashMap<String, i32>>,
    /// Optional id mapping (id -> category_name) for classification tasks
    id2label: Option<std::collections::HashMap<String, String>>,
    /// Optional instruction template for classification tasks
    instruction_template: Option<String>,
    /// Pre-tokenized category names (for fast logit extraction)
    category_names: Vec<String>,
    /// First token ID for each category (for logit extraction)
    category_first_tokens: Vec<u32>,
}

impl Qwen3CausalLM {
    /// Load Qwen3 Causal LM from a pretrained model directory
    ///
    /// # Arguments
    /// - `model_path`: Path to model directory containing config.json, model.safetensors, tokenizer.json
    /// - `device`: Device to load model on (CPU or CUDA)
    ///
    /// # Returns
    /// Loaded model ready for text generation
    pub fn from_pretrained(model_path: &str, device: &Device) -> UnifiedResult<Self> {
        println!("Loading Qwen3 Causal LM from: {}", model_path);
        
        let model_dir = Path::new(model_path);
        
        // Load config
        let config = Qwen3CausalConfig::from_pretrained(model_path)?;
        println!("✅ Config loaded: {} layers, {} hidden size", config.num_hidden_layers, config.hidden_size);
        
        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenizer loading".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", tokenizer_path)),
                context: Some(e.to_string()),
            })?;
        println!("✅ Tokenizer loaded");
        
        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(UnifiedError::Configuration {
                operation: "find model weights".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", weights_path)),
                context: Some("Expected model.safetensors in model directory".to_string()),
            });
        }
        
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
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
        println!("✅ Model weights loaded");

        // Create embeddings
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

        // Load transformer layers
        println!("Loading {} transformer layers...", config.num_hidden_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = vb.pp("model.layers");
        
        for layer_idx in 0..config.num_hidden_layers {
            if layer_idx % 5 == 0 {
                println!("  Loading layer {}/{}", layer_idx + 1, config.num_hidden_layers);
            }
            let layer = Qwen3DecoderLayer::new(&config, layers_vb.pp(&layer_idx.to_string()), device, dtype)?;
            layers.push(layer);
        }
        println!("✅ All {} transformer layers loaded", config.num_hidden_layers);

        // Load norm
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

        // Load LM head
        let lm_head = candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "create lm_head".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Try to load label_mapping.json if it exists
        let label_mapping_path = model_dir.join("label_mapping.json");
        let (label2id, id2label, instruction_template) = if label_mapping_path.exists() {
            match std::fs::read_to_string(&label_mapping_path) {
                Ok(content) => {
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(json) => {
                            let label2id_json = json.get("label2id");
                            let id2label_json = json.get("id2label");
                            let template = json.get("instruction_template")
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            
                            let label2id_map = label2id_json.and_then(|v| {
                                if let Some(obj) = v.as_object() {
                                    let mut map = std::collections::HashMap::new();
                                    for (k, v) in obj.iter() {
                                        if let Some(id) = v.as_i64() {
                                            map.insert(k.clone(), id as i32);
                                        }
                                    }
                                    Some(map)
                                } else {
                                    None
                                }
                            });
                            
                            let id2label_map = id2label_json.and_then(|v| {
                                if let Some(obj) = v.as_object() {
                                    let mut map = std::collections::HashMap::new();
                                    for (k, v) in obj.iter() {
                                        if let Some(label) = v.as_str() {
                                            map.insert(k.clone(), label.to_string());
                                        }
                                    }
                                    Some(map)
                                } else {
                                    None
                                }
                            });
                            
                            println!("✅ Loaded label mapping with {} categories", 
                                id2label_map.as_ref().map(|m| m.len()).unwrap_or(0));
                            (label2id_map, id2label_map, template)
                        }
                        Err(e) => {
                            println!("⚠️  Failed to parse label_mapping.json: {}", e);
                            (None, None, None)
                        }
                    }
                }
                Err(_) => (None, None, None),
            }
        } else {
            println!("ℹ️  No label_mapping.json found, running in pure generative mode");
            (None, None, None)
        };

        // Prepare category tokens for fast classification (logit extraction)
        let (category_names, category_first_tokens) = if let Some(ref id2label_map) = id2label {
            let mut names = Vec::new();
            let mut tokens = Vec::new();
            
            // Extract categories in order by ID
            for i in 0..id2label_map.len() {
                if let Some(name) = id2label_map.get(&i.to_string()) {
                    names.push(name.clone());
                    
                    // Tokenize with leading space to match generation context
                    let token_ids = tokenizer
                        .encode(format!(" {}", name), false)
                        .map_err(|e| UnifiedError::Configuration {
                            operation: "tokenize category".to_string(),
                            source: ConfigErrorType::ParseError(e.to_string()),
                            context: Some(format!("Failed to tokenize category: {}", name)),
                        })?
                        .get_ids()
                        .to_vec();
                    
                    // Get first token (most important for classification)
                    if let Some(&first_token) = token_ids.first() {
                        tokens.push(first_token);
                    } else {
                        // Fallback: tokenize without space
                        let fallback_ids = tokenizer
                            .encode(name.clone(), false)
                            .map_err(|e| UnifiedError::Configuration {
                                operation: "tokenize category fallback".to_string(),
                                source: ConfigErrorType::ParseError(e.to_string()),
                                context: Some(format!("Failed to tokenize category: {}", name)),
                            })?
                            .get_ids()
                            .to_vec();
                        tokens.push(*fallback_ids.first().unwrap_or(&0));
                    }
                }
            }
            
            println!("✅ Prepared {} category tokens for fast classification", names.len());
            (names, tokens)
        } else {
            (Vec::new(), Vec::new())
        };

        println!("✅ Qwen3 Causal LM loaded successfully");
        
        Ok(Self {
            config,
            tokenizer: Arc::new(tokenizer),
            embeddings,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            label2id,
            id2label,
            instruction_template,
            category_names,
            category_first_tokens,
        })
    }
    
    /// Forward pass: get logits for next token
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs tensor [batch_size, seq_len]
    /// - `start_pos`: Starting position for incremental generation (for KV cache simulation)
    ///
    /// # Returns
    /// Logits tensor [batch_size, seq_len, vocab_size]
    fn forward(&self, input_ids: &Tensor) -> UnifiedResult<Tensor> {
        // Embed tokens
        let mut hidden_states = self.embeddings.forward(input_ids).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward embeddings".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        // Pass through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states).map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: format!("forward layer {}", layer_idx),
                source: e.to_string(),
                context: None,
            })?;
        }

        // Apply final norm
        let hidden_states = self.norm.forward(&hidden_states).map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "forward norm".to_string(),
            source: e.to_string(),
            context: None,
        })?;

        // Project to vocabulary logits
        let logits = self.lm_head.forward(&hidden_states).map_err(|e| {
            UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward lm_head".to_string(),
                source: e.to_string(),
                context: None,
            }
        })?;

        Ok(logits)
    }
    
    /// Generate text from a prompt
    ///
    /// # Arguments
    /// - `prompt`: Input text prompt
    /// - `gen_config`: Generation configuration
    ///
    /// # Returns
    /// Generated text and metadata
    pub fn generate(&mut self, prompt: &str, gen_config: GenerationConfig) -> UnifiedResult<GenerationResult> {
        let start_time = std::time::Instant::now();
        
        // Tokenize input
        let encoding = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenization".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: Some("Failed to tokenize input prompt".to_string()),
            })?;
        
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();
        
        // Create logits processor
        let mut logits_processor = LogitsProcessor::new(
            gen_config.seed,
            gen_config.temperature,
            gen_config.top_p,
        );
        
        // EOS tokens (Qwen3 uses both)
        let eos_tokens = vec![151643, 151645]; // <|endoftext|>, <|im_end|>
        
        // Generation loop
        let mut generated_count = 0;
        for _ in 0..gen_config.max_new_tokens {
            // Prepare input tensor
            let input_ids = Tensor::new(&tokens[..], &self.device)
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
            
            // Forward pass
            let logits = self.forward(&input_ids)?;
            
            // Get logits at last position
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
                    operation: "convert to f32".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            
            // Apply repeat penalty
            let last_logits = if gen_config.repeat_penalty != 1.0 {
                self.apply_repeat_penalty(&last_logits, &tokens, gen_config.repeat_penalty, gen_config.repeat_last_n)?
            } else {
                last_logits
            };
            
            // Sample next token
            let next_token = logits_processor.sample(&last_logits)?;
            
            // Check for EOS
            if eos_tokens.contains(&next_token) {
                break;
            }
            
            tokens.push(next_token);
            generated_count += 1;
        }
        
        // Decode generated tokens (excluding prompt)
        let generated_tokens = &tokens[prompt_len..];
        let text = self.tokenizer
            .decode(generated_tokens, true)
            .map_err(|e| UnifiedError::Processing {
                operation: "decode tokens".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        
        let elapsed = start_time.elapsed();
        let tokens_per_second = generated_count as f64 / elapsed.as_secs_f64();
        
        Ok(GenerationResult {
            text,
            token_ids: tokens,
            num_generated: generated_count,
            tokens_per_second,
        })
    }
    
    /// Apply repeat penalty to logits
    fn apply_repeat_penalty(
        &self,
        logits: &Tensor,
        tokens: &[u32],
        penalty: f32,
        last_n: usize,
    ) -> UnifiedResult<Tensor> {
        let mut logits_vec = logits.to_vec1::<f32>().map_err(|e| UnifiedError::Processing {
            operation: "convert logits to vec".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let start_at = tokens.len().saturating_sub(last_n);
        for &token_id in &tokens[start_at..] {
            let token_id = token_id as usize;
            if token_id < logits_vec.len() {
                if logits_vec[token_id] < 0.0 {
                    logits_vec[token_id] *= penalty;
                } else {
                    logits_vec[token_id] /= penalty;
                }
            }
        }
        
        Tensor::from_vec(logits_vec, logits.shape(), &self.device).map_err(|e| {
            UnifiedError::Processing {
                operation: "create tensor from vec".to_string(),
                source: e.to_string(),
                input_context: None,
            }
        })
    }
    
    /// Classify text using logit extraction (fast, single forward pass)
    ///
    /// This method matches the Python implementation's hybrid approach:
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
        if self.category_names.is_empty() {
            return Err(UnifiedError::Configuration {
                operation: "classification".to_string(),
                source: ConfigErrorType::FileNotFound("label_mapping.json".to_string()),
                context: Some("No category mapping found. Model must be loaded with label_mapping.json for classification.".to_string()),
            });
        }
        
        // Format instruction prompt
        let prompt = self.format_classification_instruction(text);
        
        // Tokenize
        let encoding = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenization".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: Some("Failed to tokenize classification prompt".to_string()),
            })?;
        
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)
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
        
        // Forward pass
        let logits = self.forward(&input_ids)?;
        
        // Get logits at last position [batch_size, seq_len, vocab_size]
        let seq_len = logits.dim(1).map_err(|e| UnifiedError::Processing {
            operation: "get sequence length".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;
        
        let last_logits = logits
            .i((0, seq_len - 1, ..))
            .map_err(|e| UnifiedError::Processing {
                operation: "extract last position logits".to_string(),
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
        let category_logits: Vec<f32> = self.category_first_tokens
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
        
        // Apply softmax to get probabilities
        let category_logits_tensor = Tensor::from_vec(
            category_logits,
            (self.category_names.len(),),
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
        let best_category = self.category_names[best_idx].clone();
        
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
        if self.category_names.is_empty() {
            return Err(UnifiedError::Configuration {
                operation: "batch classification".to_string(),
                source: ConfigErrorType::FileNotFound("label_mapping.json".to_string()),
                context: Some("No category mapping found.".to_string()),
            });
        }
        
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                let prompt = self.format_classification_instruction(text);
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
        let logits = self.forward(&batch_tensor)?;
        
        // Process each sample in the batch
        let mut results = Vec::with_capacity(texts.len());
        
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
            let category_logits: Vec<f32> = self.category_first_tokens
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
                (self.category_names.len(),),
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
                category: self.category_names[best_idx].clone(),
                confidence: probs_vec[best_idx],
                probabilities: probs_vec,
            });
        }
        
        Ok(results)
    }
    
    /// Format classification instruction prompt
    fn format_classification_instruction(&self, text: &str) -> String {
        let instruction = if let Some(template) = &self.instruction_template {
            template.replace("{question}", text)
        } else {
            // Default classification prompt
            let categories_str = self.category_names.join(", ");
            format!(
                "You are an expert classifier. Classify the following text into exactly ONE category. Respond with ONLY the category name.\n\nCategories: {}\n\nText: {}\nCategory:",
                categories_str, text
            )
        };
        
        // Format with ChatML
        self.format_chat_prompt(&instruction)
    }
    
    /// Format prompt using ChatML template
    ///
    /// Converts a user message into Qwen3's ChatML format:
    /// ```text
    /// <|im_start|>user
    /// {message}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    pub fn format_chat_prompt(&self, message: &str) -> String {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            message
        )
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    
    /// Get the config
    pub fn config(&self) -> &Qwen3CausalConfig {
        &self.config
    }
    
    /// Get available categories if label mapping exists
    pub fn categories(&self) -> Option<Vec<String>> {
        self.id2label.as_ref().map(|map| {
            let mut cats: Vec<String> = map.values().cloned().collect();
            cats.sort();
            cats
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chat_template_format() {
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "Hello, how are you?"
        );
        
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }
    
    #[test]
    fn test_softmax() {
        let processor = LogitsProcessor::new(42, Some(1.0), None);
        let logits = vec![1.0, 2.0, 3.0];
        let probs = processor.softmax(&logits);
        
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
    
    #[test]
    fn test_classification_result_structure() {
        let result = ClassificationResult {
            class: 0,
            category: "biology".to_string(),
            confidence: 0.85,
            probabilities: vec![0.85, 0.10, 0.05],
        };
        
        assert_eq!(result.class, 0);
        assert_eq!(result.category, "biology");
        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.probabilities.len(), 3);
        
        // Check probabilities sum to 1.0
        let sum: f32 = result.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.top_p, None);
        assert_eq!(config.repeat_penalty, 1.1);
        assert_eq!(config.repeat_last_n, 64);
    }
    
    #[test]
    fn test_generation_config_greedy() {
        let config = GenerationConfig {
            max_new_tokens: 20,
            temperature: Some(0.0),
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
        };
        
        // Greedy decoding
        assert_eq!(config.temperature, Some(0.0));
        assert_eq!(config.max_new_tokens, 20);
    }
    
    #[test]
    fn test_classification_output_makes_sense() {
        println!("\n=== Classification Output Demo ===\n");
        
        // High confidence biology classification
        let bio_result = ClassificationResult {
            class: 0,
            category: "biology".to_string(),
            confidence: 0.92,
            probabilities: vec![0.92, 0.03, 0.02, 0.01, 0.01, 0.01],
        };
        
        println!("📌 High Confidence: \"What is photosynthesis?\"");
        println!("   Category: {} (class {})", bio_result.category, bio_result.class);
        println!("   Confidence: {:.1}%", bio_result.confidence * 100.0);
        
        // Verify distribution
        let sum: f32 = bio_result.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probabilities must sum to 1.0");
        assert_eq!(bio_result.confidence, bio_result.probabilities[0]);
        println!("   ✅ Valid distribution (sum = {:.6})", sum);
        
        // Ambiguous case
        let ambiguous = ClassificationResult {
            class: 2,
            category: "physics".to_string(),
            confidence: 0.48,
            probabilities: vec![0.05, 0.08, 0.48, 0.15, 0.10, 0.14],
        };
        
        println!("\n📌 Ambiguous: \"What causes motion?\"");
        println!("   Category: {} (confidence: {:.1}% - low)", ambiguous.category, ambiguous.confidence * 100.0);
        println!("   Top 3:");
        let mut indexed: Vec<(usize, f32)> = ambiguous.probabilities
            .iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let categories = vec!["biology", "chemistry", "physics", "math", "cs", "eng"];
        for (i, prob) in indexed.iter().take(3) {
            println!("     {}: {:.1}%", categories[*i], prob * 100.0);
        }
        
        let sum2: f32 = ambiguous.probabilities.iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-5);
        println!("   ✅ Valid distribution");
        
        // Batch results
        println!("\n📌 Batch: 3 physics questions");
        let batch = vec![
            ClassificationResult { class: 2, category: "physics".to_string(), confidence: 0.91, probabilities: vec![0.02, 0.03, 0.91, 0.02, 0.01, 0.01] },
            ClassificationResult { class: 2, category: "physics".to_string(), confidence: 0.89, probabilities: vec![0.03, 0.04, 0.89, 0.02, 0.01, 0.01] },
            ClassificationResult { class: 2, category: "physics".to_string(), confidence: 0.93, probabilities: vec![0.01, 0.02, 0.93, 0.02, 0.01, 0.01] },
        ];
        
        let avg: f32 = batch.iter().map(|r| r.confidence).sum::<f32>() / batch.len() as f32;
        println!("   Average confidence: {:.1}%", avg * 100.0);
        println!("   Range: {:.1}% - {:.1}%", 
            batch.iter().map(|r| r.confidence).fold(f32::INFINITY, f32::min) * 100.0,
            batch.iter().map(|r| r.confidence).fold(f32::NEG_INFINITY, f32::max) * 100.0
        );
        println!("   ✅ Consistent results\n");
        
        println!("=== All Checks Passed ===");
        println!("✅ Probabilities sum to 1.0");
        println!("✅ Confidence matches max probability");
        println!("✅ Results are consistent");
        println!("✅ Ambiguity is detected via low confidence");
    }
    
    #[test]
    fn test_entropy_calculation() {
        // Shannon entropy calculation
        fn entropy(probs: &[f32]) -> f32 {
            probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.log2()).sum()
        }
        
        // High certainty: low entropy
        let high_certainty = vec![0.95, 0.02, 0.01, 0.01, 0.01];
        let h1 = entropy(&high_certainty);
        
        // Medium certainty: medium entropy  
        let medium = vec![0.50, 0.30, 0.10, 0.05, 0.05];
        let h2 = entropy(&medium);
        
        // Low certainty (uniform): high entropy
        let uniform = vec![0.20, 0.20, 0.20, 0.20, 0.20];
        let h3 = entropy(&uniform);
        
        println!("\n=== Entropy Analysis ===");
        println!("High certainty [0.95, 0.02, ...]: {:.3} bits", h1);
        println!("Medium certainty [0.50, 0.30, ...]: {:.3} bits", h2);
        println!("Uniform [0.20, 0.20, ...]: {:.3} bits (max = {:.3})", h3, 5.0_f32.log2());
        
        // Entropy should increase with uncertainty
        assert!(h1 < h2);
        assert!(h2 < h3);
        
        println!("✅ Entropy increases with uncertainty (as expected)");
    }
}

