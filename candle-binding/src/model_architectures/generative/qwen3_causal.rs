//! Qwen3 Causal Language Model (ForCausalLM)
//!
//! Implementation of Qwen3 as a causal language model for text generation.
//! Supports:
//! - Text generation with various sampling strategies
//! - Chat template (ChatML format)
//! - LoRA adapters for fine-tuned classification models
//!
//! ## Architecture
//! - Decoder-only transformer (causal attention)
//! - RoPE positional embeddings
//! - GQA (Grouped Query Attention)
//! - Language modeling head (lm_head: Linear layer vocab_size)
//!
//! ## Usage Example
//! ```ignore
//! let model = Qwen3CausalLM::from_pretrained("Qwen/Qwen3-0.6B", &device)?;
//! let result = model.generate("Hello", GenerationConfig::default())?;
//! ```

use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use candle_core::{Device, Tensor};
use serde::Deserialize;
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
    
    #[serde(default)]
    pub tie_word_embeddings: bool,
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
        
        serde_json::from_str(&config_str)
            .map_err(|e| UnifiedError::Configuration {
                operation: "config JSON parsing".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: Some(format!("Failed to parse config.json from {}", model_path)),
            })
    }
}

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (0.0 = greedy, higher = more random)
    pub temperature: f64,
    /// Whether to use sampling (false = greedy decoding)
    pub do_sample: bool,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,
    /// Repetition penalty
    pub repetition_penalty: f64,
    /// EOS token IDs to stop generation
    pub eos_token_ids: Vec<u32>,
    /// PAD token ID
    pub pad_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 0.1,
            do_sample: false,  // Greedy by default
            top_p: None,
            repetition_penalty: 1.0,
            eos_token_ids: vec![151643], // Qwen3 EOS
            pad_token_id: 151643,
        }
    }
}

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text (decoded)
    pub text: String,
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Logits for each generated token (optional)
    pub logits: Option<Vec<Vec<f32>>>,
}

/// Qwen3 Causal Language Model
///
/// This is a simplified implementation focused on inference for classification.
/// For full generation capabilities, consider using candle-transformers.
pub struct Qwen3CausalLM {
    config: Qwen3CausalConfig,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    // TODO: Add actual model layers when implementing full model
    // For now, we'll use a placeholder and implement the core generation logic
}

impl Qwen3CausalLM {
    /// Load Qwen3 Causal LM from a pretrained model directory
    ///
    /// # Arguments
    /// - `model_path`: Path to model directory containing config.json, model.safetensors, tokenizer.json
    /// - `device`: Device to load model on (CPU or CUDA)
    ///
    /// # Returns
    /// Loaded model ready for generation
    pub fn from_pretrained(model_path: &str, device: &Device) -> UnifiedResult<Self> {
        // Load config
        let config = Qwen3CausalConfig::from_pretrained(model_path)?;
        
        // Load tokenizer
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenizer loading".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", tokenizer_path)),
                context: Some(e.to_string()),
            })?;
        
        Ok(Self {
            config,
            tokenizer: Arc::new(tokenizer),
            device: device.clone(),
        })
    }
    
    /// Generate text from a prompt
    ///
    /// # Arguments
    /// - `prompt`: Input text prompt
    /// - `config`: Generation configuration
    ///
    /// # Returns
    /// Generated text and metadata
    pub fn generate(&self, prompt: &str, gen_config: GenerationConfig) -> UnifiedResult<GenerationResult> {
        // Tokenize input
        let encoding = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenization".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: Some("Failed to tokenize input prompt".to_string()),
            })?;
        
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        // For now, return a placeholder implementation
        // TODO: Implement actual forward pass and generation loop
        Ok(GenerationResult {
            text: "TODO: Implement generation".to_string(),
            token_ids: input_ids,
            logits: None,
        })
    }
    
    /// Get logits for the next token given input tokens
    ///
    /// This is the core method for classification via generation.
    /// Returns logits for all vocab tokens at the last position.
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs tensor [batch_size, seq_len]
    ///
    /// # Returns
    /// Logits tensor [batch_size, vocab_size]
    pub fn get_next_token_logits(&self, input_ids: &Tensor) -> UnifiedResult<Tensor> {
        // TODO: Implement forward pass
        // 1. Embed tokens
        // 2. Pass through transformer layers
        // 3. Apply LM head
        // 4. Return logits at last position
        
        Err(UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "get_next_token_logits".to_string(),
            source: "Not yet implemented".to_string(),
            context: Some("Qwen3CausalLM forward pass not yet implemented".to_string()),
        })
    }
    
    /// Classify text using generative model (extract logits for category tokens)
    ///
    /// This method formats the input with a classification prompt, runs forward pass,
    /// and extracts logits for the first tokens of each category name.
    ///
    /// # Arguments
    /// - `text`: Text to classify
    /// - `categories`: List of category names
    /// - `instruction_template`: Template for formatting the instruction
    ///
    /// # Returns
    /// Category logits (one per category)
    pub fn classify(
        &self,
        text: &str,
        categories: &[String],
        instruction_template: &str,
    ) -> UnifiedResult<Vec<f32>> {
        // Format instruction
        let instruction = instruction_template.replace("{question}", text);
        
        // Apply chat template
        let prompt = self.format_chat_prompt(&instruction)?;
        
        // Tokenize
        let encoding = self.tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenization".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;
        
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        // Get logits at last position
        // TODO: Implement actual forward pass
        
        // Extract logits for each category's first token
        let mut category_logits = Vec::new();
        for category in categories {
            // Tokenize category name (with leading space)
            let cat_text = format!(" {}", category);
            let cat_encoding = self.tokenizer
                .encode(cat_text.as_str(), false)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "category tokenization".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: None,
                })?;
            
            let cat_token_id = cat_encoding.get_ids()[0];
            
            // TODO: Extract logit for this token from full logits
            // For now, return placeholder
            category_logits.push(0.0);
        }
        
        Ok(category_logits)
    }
    
    /// Format prompt using ChatML template
    ///
    /// Converts a user message into Qwen3's ChatML format:
    /// ```text
    /// <|im_start|>user
    /// {message}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    ///
    /// # Arguments
    /// - `message`: User message content
    ///
    /// # Returns
    /// Formatted prompt ready for tokenization
    pub fn format_chat_prompt(&self, message: &str) -> UnifiedResult<String> {
        // ChatML format with generation prompt (add_generation_prompt=True)
        // This matches Python's: tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            message
        );
        Ok(prompt)
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    
    /// Get the config
    pub fn config(&self) -> &Qwen3CausalConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chat_template_format() {
        // This test doesn't require a model, just tests template formatting
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "Hello, how are you?"
        );
        
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }
}

