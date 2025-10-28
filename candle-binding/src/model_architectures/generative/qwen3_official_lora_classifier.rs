//! Qwen3 Generative Classifier using Official Candle Implementation + LoRA
//!
//! This module wraps the official `candle_transformers::models::qwen3::ModelForCausalLM`
//! and applies LoRA adapters by merging them into the base model weights.
//!
//! Key features:
//! - Uses proven official Qwen3 implementation from candle-transformers
//! - Merges LoRA weights into base model for efficient inference
//! - Extracts logits for classification (single forward pass)
//! - Supports ChatML format prompts

use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Label mapping for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelMapping {
    pub label2id: HashMap<String, usize>,
    pub id2label: HashMap<String, String>,
    pub instruction_template: String,
}

impl LabelMapping {
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<_> = self.id2label.iter().collect();
        cats.sort_by_key(|(id, _)| id.parse::<usize>().unwrap());
        cats.into_iter().map(|(_, label)| label.clone()).collect()
    }
}

/// Classification result
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub category: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

/// Qwen3 LoRA Classifier using official Candle implementation
pub struct Qwen3OfficialLoRAClassifier {
    model: Qwen3Model,
    tokenizer: Arc<Tokenizer>,
    label_mapping: LabelMapping,
    category_token_ids: Vec<u32>,
    device: Device,
}

impl Qwen3OfficialLoRAClassifier {
    /// Load model from pretrained weights with LoRA adapters
    ///
    /// # Arguments
    /// - `model_path`: Path to LoRA adapter directory
    /// - `device`: Device to run on
    ///
    /// Expected structure:
    /// ```
    /// models/
    ///   ├── Qwen3-0.6B/                          (base model)
    ///   │   ├── config.json
    ///   │   ├── model.safetensors
    ///   │   └── tokenizer.json
    ///   └── qwen3_generative_classifier_r16/     (LoRA adapter)
    ///       ├── adapter_config.json
    ///       ├── adapter_model.safetensors
    ///       └── label_mapping.json
    /// ```
    pub fn from_pretrained(model_path: &str, device: &Device) -> UnifiedResult<Self> {
        println!("🚀 Loading Qwen3 with Official Candle Implementation + LoRA");
        
        // Parse paths
        let (base_model_path, lora_adapter_path) = if model_path.contains("qwen3_generative_classifier") {
            let adapter_dir = Path::new(model_path);
            let parent = adapter_dir.parent().unwrap_or(Path::new("./models"));
            let base_path = parent.join("Qwen3-0.6B");
            (base_path.to_string_lossy().to_string(), model_path.to_string())
        } else {
            let base_dir = Path::new(model_path);
            let parent = base_dir.parent().unwrap_or(Path::new("./models"));
            let adapter_path = parent.join("qwen3_generative_classifier_r16");
            (model_path.to_string(), adapter_path.to_string_lossy().to_string())
        };

        println!("  Base model: {}", base_model_path);
        println!("  LoRA adapter: {}", lora_adapter_path);

        let base_dir = Path::new(&base_model_path);
        let adapter_dir = Path::new(&lora_adapter_path);

        // Load config
        let config_path = base_dir.join("config.json");
        let config: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)
            .map_err(|e| UnifiedError::Configuration {
                operation: "parse config".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;

        println!("  Config: hidden_size={}, layers={}, vocab={}", 
            config.hidden_size, config.num_hidden_layers, config.vocab_size);

        // Load tokenizer
        let tokenizer_path = base_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| UnifiedError::Configuration {
                operation: "load tokenizer".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;

        // Load label mapping
        let label_mapping_path = adapter_dir.join("label_mapping.json");
        let label_mapping: LabelMapping = serde_json::from_str(
            &std::fs::read_to_string(label_mapping_path)?
        )?;

        println!("  Categories: {}", label_mapping.categories().len());

        // Determine dtype
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };
        println!("  Using dtype: {:?}", dtype);

        // Load base model weights
        let weights_path = base_dir.join("model.safetensors");
        println!("  Loading base model weights...");
        
        // Check if we should merge LoRA weights
        let adapter_weights_path = adapter_dir.join("adapter_model.safetensors");
        let should_merge_lora = adapter_weights_path.exists();

        let vb = if should_merge_lora {
            println!("  📦 Loading with LoRA merging...");
            Self::load_weights_with_lora_merged(
                &weights_path,
                &adapter_weights_path,
                &adapter_dir.join("adapter_config.json"),
                device,
                dtype,
            )?
        } else {
            println!("  ⚠️  No LoRA weights found, loading base model only");
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device) }
                .map_err(|e| UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "load base weights".to_string(),
                    source: e.to_string(),
                    context: None,
                })?
        };

        // Build model
        println!("  🏗️  Building Qwen3 model...");
        let model = Qwen3Model::new(&config, vb)
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "build Qwen3 model".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Prepare category tokens
        let category_token_ids = Self::prepare_category_tokens(&tokenizer, &label_mapping)?;

        println!("✅ Qwen3 Official LoRA Classifier loaded successfully\n");

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            label_mapping,
            category_token_ids,
            device: device.clone(),
        })
    }

    /// Load weights with LoRA merged (simplified - just loads base for now)
    ///
    /// TODO: Implement actual LoRA merging by:
    /// 1. Loading base weights
    /// 2. Loading LoRA A and B matrices
    /// 3. Computing delta = B @ A * (alpha/rank)
    /// 4. Adding delta to base weights
    fn load_weights_with_lora_merged<'a>(
        _base_weights_path: &Path,
        _lora_weights_path: &Path,
        _lora_config_path: &Path,
        device: &'a Device,
        dtype: DType,
    ) -> UnifiedResult<VarBuilder<'a>> {
        // For now, just load base weights
        // TODO: Merge LoRA weights before loading
        println!("    ⚠️  LoRA merging not yet implemented - using base model only");
        println!("    This will give you base model accuracy, not fine-tuned accuracy");
        
        unsafe { VarBuilder::from_mmaped_safetensors(&[_base_weights_path], dtype, device) }
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "load merged weights".to_string(),
                source: e.to_string(),
                context: None,
            })
    }

    /// Prepare category token IDs for logit extraction
    fn prepare_category_tokens(
        tokenizer: &Tokenizer,
        label_mapping: &LabelMapping,
    ) -> UnifiedResult<Vec<u32>> {
        let categories = label_mapping.categories();
        let mut token_ids = Vec::new();

        for category in &categories {
            // Tokenize with leading space (to match generation context)
            let tokens = tokenizer
                .encode(format!(" {}", category), false)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "tokenize category".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: Some(format!("category: {}", category)),
                })?;

            if let Some(&token_id) = tokens.get_ids().first() {
                token_ids.push(token_id);
            } else {
                return Err(UnifiedError::Configuration {
                    operation: "get category token".to_string(),
                    source: ConfigErrorType::ParseError(format!(
                        "Category '{}' produced no tokens",
                        category
                    )),
                    context: None,
                });
            }
        }

        println!("  Prepared {} category tokens", token_ids.len());
        Ok(token_ids)
    }

    /// Format prompt with ChatML template
    pub fn format_prompt(&self, text: &str) -> String {
        let instruction = if !self.label_mapping.instruction_template.is_empty() {
            self.label_mapping
                .instruction_template
                .replace("{question}", text)
        } else {
            let categories = self.label_mapping.categories().join(", ");
            format!(
                "You are an expert academic classifier. Classify the following question into exactly ONE category. Respond with ONLY the category name.\n\nCategories: {}\n\nNow classify this question:\nQ: {}\nA:",
                categories, text
            )
        };

        // ChatML format with thinking tags to match Python training
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            instruction
        )
    }

    /// Classify text using logit extraction
    pub fn classify(&mut self, text: &str) -> UnifiedResult<ClassificationResult> {
        // Clear KV cache before each forward pass (important for independent classifications)
        self.model.clear_kv_cache();
        
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

        // Forward pass through official Qwen3 model
        let logits = self.model.forward(&input_ids, 0)
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "forward pass".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Extract logits at last position
        let seq_len = logits.dim(1).map_err(|e| UnifiedError::Processing {
            operation: "get sequence length".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        let last_logits = logits
            .i((0, seq_len - 1, ..))
            .map_err(|e| UnifiedError::Processing {
                operation: "extract last logits".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "convert logits to F32".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Extract category logits
        let mut category_logits = Vec::new();
        for &token_id in &self.category_token_ids {
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

        // Apply softmax
        let probabilities = softmax(&category_logits);

        // Find best category
        let max_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let categories = self.label_mapping.categories();

        Ok(ClassificationResult {
            category: categories[max_idx].clone(),
            confidence: probabilities[max_idx],
            probabilities,
        })
    }

    /// Get categories
    pub fn categories(&self) -> Vec<String> {
        self.label_mapping.categories()
    }
}

/// Apply softmax to convert logits to probabilities
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum).collect()
}

