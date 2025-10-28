use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Config3, ModelForCausalLM as Model3};
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    println!("Testing Official Candle Qwen3 Implementation");
    println!("{}", "=".repeat(70));
    
    let model_path = "/home/ubuntu/rootfs/back/semantic-router.bak/models/Qwen3-0.6B";
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::BF16;
    
    println!("\n📥 Loading config...");
    let config_file = format!("{}/config.json", model_path);
    let config: Config3 = serde_json::from_slice(&std::fs::read(config_file)?)?;
    println!("   Config: hidden_size={}, layers={}, vocab={}", 
        config.hidden_size, config.num_hidden_layers, config.vocab_size);
    
    println!("\n📥 Loading model weights...");
    let weights_path = format!("{}/model.safetensors", model_path);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
    
    println!("\n🏗️  Building model...");
    let mut model = Model3::new(&config, vb)?;
    println!("✅ Model loaded successfully");
    
    println!("\n📥 Loading tokenizer...");
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Load categories
    let adapter_path = "/home/ubuntu/rootfs/back/semantic-router.bak/models/qwen3_generative_classifier_r16_fixed";
    let label_mapping: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/label_mapping.json", adapter_path))?
    )?;
    
    let categories: Vec<String> = (0..14)
        .map(|i| label_mapping["id2label"][i.to_string()].as_str().unwrap().to_string())
        .collect();
    
    // Get category token IDs
    let category_token_ids: Vec<u32> = categories
        .iter()
        .map(|cat| {
            let tokens = tokenizer.encode(format!(" {}", cat), false)
                .expect(&format!("Failed to encode category: {}", cat));
            tokens.get_ids()[0]
        })
        .collect();
    
    println!("\n📝 Testing prompt:");
    let instruction_template = label_mapping["instruction_template"].as_str().unwrap();
    let test_text = "What does GDP measure in an economy?";
    let instruction = instruction_template.replace("{question}", test_text);
    
    // Format with chat template (with thinking tags to match Python)
    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        instruction
    );
    
    println!("Text: {}", test_text);
    
    // Tokenize
    let encoding = tokenizer.encode(prompt.clone(), true)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
    let tokens = encoding.get_ids();
    println!("Tokens: {} (prompt length)", tokens.len());
    
    // Convert to tensor
    let input_ids = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    
    println!("\n🔄 Running forward pass...");
    let logits = model.forward(&input_ids, 0)?;
    
    // Get last token logits
    let seq_len = logits.dim(1)?;
    let last_logits = logits.i((0, seq_len - 1, ..))?;
    let last_logits_f32 = last_logits.to_dtype(DType::F32)?;
    
    // Extract category logits
    println!("\n📊 Category Logits (Official Candle):");
    let mut category_logits = Vec::new();
    for (i, (cat, &token_id)) in categories.iter().zip(category_token_ids.iter()).enumerate() {
        let logit = last_logits_f32.i(token_id as usize)?.to_scalar::<f32>()?;
        category_logits.push(logit);
        println!("  [{:2}] {:20}: {:8.3}", i, cat, logit);
    }
    
    // Compute softmax
    let max_logit = category_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = category_logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();
    
    let max_idx = probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    println!("\n🎯 Prediction: {} ({:.2}%)", categories[max_idx], probs[max_idx] * 100.0);
    println!("\nTop 5 predictions:");
    let mut sorted: Vec<_> = probs.iter().enumerate().collect();
    sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    for (idx, prob) in sorted.iter().take(5) {
        println!("  {:20}: {:6.2}%", categories[*idx], *prob * 100.0);
    }
    
    Ok(())
}

