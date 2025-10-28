use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_lora_classifier::Qwen3LoRAClassifier;
use candle_semantic_router::core::UnifiedResult;

fn main() -> UnifiedResult<()> {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("Loading model...");
    let model = Qwen3LoRAClassifier::from_pretrained(&model_path, &device)?;
    
    let test_text = "What is GDP?";
    let prompt = model.format_prompt(test_text);
    
    println!("\n=== PROMPT ===");
    println!("{}", prompt);
    println!("\n=== TOKENIZATION WITH add_special_tokens=true ===");
    let encoding_true = model.tokenizer().encode(prompt.as_str(), true)
        .map_err(|e| candle_semantic_router::core::UnifiedError::Configuration {
            operation: "tokenize".to_string(),
            source: candle_semantic_router::core::ConfigErrorType::ParseError(e.to_string()),
            context: None,
        })?;
    println!("Token IDs ({} tokens): {:?}", encoding_true.len(), &encoding_true.get_ids()[..encoding_true.len().min(50)]);
    println!("First 10 tokens: {:?}", &encoding_true.get_tokens()[..10.min(encoding_true.len())]);
    
    println!("\n=== TOKENIZATION WITH add_special_tokens=false ===");
    let encoding_false = model.tokenizer().encode(prompt.as_str(), false)
        .map_err(|e| candle_semantic_router::core::UnifiedError::Configuration {
            operation: "tokenize".to_string(),
            source: candle_semantic_router::core::ConfigErrorType::ParseError(e.to_string()),
            context: None,
        })?;
    println!("Token IDs ({} tokens): {:?}", encoding_false.len(), &encoding_false.get_ids()[..encoding_false.len().min(50)]);
    println!("First 10 tokens: {:?}", &encoding_false.get_tokens()[..10.min(encoding_false.len())]);
    
    println!("\n=== CATEGORY TOKENS ===");
    for category in model.categories() {
        let with_space = format!(" {}", category);
        let tokens_with_space = model.tokenizer().encode(with_space.as_str(), false)
            .map_err(|e| candle_semantic_router::core::UnifiedError::Configuration {
                operation: "tokenize".to_string(),
                source: candle_semantic_router::core::ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;
        println!("Category '{}': first token = {}", category, tokens_with_space.get_ids()[0]);
    }
    
    Ok(())
}

