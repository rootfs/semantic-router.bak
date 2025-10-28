//! Generative Model Architectures
//!
//! This module contains implementations of causal language models (decoder-only)
//! for text generation tasks.
//!
//! ## Models
//! - **Qwen3**: Qwen3ForCausalLM for text generation and classification via generation
//! - **Qwen3LoRAClassifier**: Optimized classifier using Qwen3 + LoRA for category classification
//!
//! ## Features
//! - Text generation with sampling strategies
//! - Chat template support (ChatML format)
//! - LoRA adapter support for fine-tuned models
//! - Greedy and temperature-based sampling
//! - Generative classification (extract logits for category tokens)

pub mod qwen3_causal;
pub mod qwen3_lora_classifier;
pub mod qwen3_official_lora_classifier;
pub mod qwen3_multi_lora_classifier;

pub use qwen3_causal::{Qwen3CausalLM, Qwen3CausalConfig, GenerationConfig, GenerationResult};
pub use qwen3_lora_classifier::{Qwen3LoRAClassifier, LabelMapping, softmax};
pub use qwen3_official_lora_classifier::{Qwen3OfficialLoRAClassifier, ClassificationResult};
pub use qwen3_multi_lora_classifier::{Qwen3MultiLoRAClassifier, MultiAdapterClassificationResult};
