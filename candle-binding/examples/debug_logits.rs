use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_lora_classifier::Qwen3LoRAClassifier;
use candle_semantic_router::core::UnifiedResult;

fn main() -> UnifiedResult<()> {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("Loading model...");
    let model = Qwen3LoRAClassifier::from_pretrained(&model_path, &device)?;
    
    let test_texts = vec![
        ("What does GDP measure in an economy?", "economics"),
        ("What is Ohm's Law in electrical engineering?", "engineering"),
        ("In marketing, what are the four Ps of the marketing mix?", "business"),
    ];
    
    for (text, expected) in test_texts {
        println!("\n{}", "=".repeat(70));
        println!("Text: {}", text);
        println!("Expected: {}", expected);
        println!("{}", "-".repeat(70));
        
        let result = model.classify(text)?;
        
        println!("Predicted: {} ({:.2}%)", result.category, result.confidence * 100.0);
        println!("\nRaw logits and probabilities:");
        
        // Print sorted by probability
        let mut cat_probs: Vec<_> = result.probabilities.iter().enumerate().collect();
        cat_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        for (idx, prob) in cat_probs.iter().take(5) {
            let category = &model.categories()[*idx];
            println!("  {:20}: prob={:6.2}%", category, *prob * 100.0);
        }
    }
    
    Ok(())
}

