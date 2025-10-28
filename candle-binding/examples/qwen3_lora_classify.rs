use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_lora_classifier::Qwen3LoRAClassifier;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Qwen3 LoRA Classifier Demo ===\n");
    
    // Set model path
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16".to_string());
    
    println!("Loading model from: {}", model_path);
    
    // Initialize device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}\n", device);
    
    // Load model
    let model = Qwen3LoRAClassifier::from_pretrained(&model_path, &device)?;
    
    println!("\n✅ Model loaded successfully");
    println!("Categories: {:?}\n", model.categories());
    
    // Test samples
    let test_samples = vec![
        "What is photosynthesis?",
        "Explain the quicksort algorithm",
        "What is Newton's third law?",
        "How does GDP affect inflation?",
    ];
    
    println!("=== Single Classification Tests ===\n");
    for (i, text) in test_samples.iter().enumerate() {
        println!("Sample {}: \"{}\"", i + 1, text);
        
        let result = model.classify(text)?;
        
        println!("  Category: {} (class {})", result.category, result.class);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!("  Top 3 probabilities:");
        
        let mut indexed: Vec<(usize, f32)> = result.probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let categories = model.categories();
        for (idx, prob) in indexed.iter().take(3) {
            println!("    {}: {:.2}%", categories[*idx], prob * 100.0);
        }
        println!();
    }
    
    println!("=== Batch Classification Test ===\n");
    let batch_texts: Vec<&str> = test_samples.iter().map(|s| s.as_ref()).collect();
    let batch_results = model.classify_batch(&batch_texts)?;
    
    println!("Batch classification of {} samples:", batch_texts.len());
    for (i, result) in batch_results.iter().enumerate() {
        println!("  {}. {} -> {} ({:.2}%)", 
            i + 1, 
            batch_texts[i], 
            result.category, 
            result.confidence * 100.0
        );
    }
    
    println!("\n✅ All tests completed successfully!");
    
    Ok(())
}

