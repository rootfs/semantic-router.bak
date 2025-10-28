//! Simple example: Testing Qwen3 Classification
//!
//! This is a minimal example to verify the logit extraction classification works.
//! 
//! Run with:
//! ```bash
//! cargo run --example qwen3_classify_simple --features cuda
//! ```

use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_causal::Qwen3CausalLM;

fn main() -> anyhow::Result<()> {
    println!("🧪 Qwen3 Classification Test\n");
    
    // Configuration (hardcoded for testing)
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "./models/qwen3_generative_classifier_r16".to_string());
    
    let use_cpu = std::env::var("USE_CPU").is_ok();
    
    // Setup device
    let device = if use_cpu {
        println!("📌 Using CPU");
        Device::Cpu
    } else {
        match Device::cuda_if_available(0) {
            Ok(dev) if dev.is_cuda() => {
                println!("📌 Using CUDA GPU");
                dev
            }
            _ => {
                println!("📌 CUDA not available, using CPU");
                Device::Cpu
            }
        }
    };
    
    println!("📂 Loading model from: {}", model_path);
    
    // Load model
    let model = match Qwen3CausalLM::from_pretrained(&model_path, &device) {
        Ok(m) => {
            println!("✅ Model loaded successfully\n");
            m
        }
        Err(e) => {
            eprintln!("❌ Failed to load model: {:?}", e);
            eprintln!("\n💡 Note: This example requires a trained model.");
            eprintln!("Set MODEL_PATH environment variable to your model directory.");
            eprintln!("\nExample:");
            eprintln!("  MODEL_PATH=./path/to/model cargo run --example qwen3_classify_simple");
            return Ok(());
        }
    };
    
    // Show available categories
    if let Some(categories) = model.categories() {
        println!("📚 Available categories ({}):", categories.len());
        for (i, cat) in categories.iter().enumerate() {
            println!("  [{}] {}", i, cat);
        }
        println!();
    } else {
        println!("⚠️  No categories found (missing label_mapping.json)\n");
        return Ok(());
    }
    
    // Test texts
    let test_texts = vec![
        "What is photosynthesis?",
        "How do I calculate compound interest?",
        "What causes earthquakes?",
    ];
    
    println!("🔬 Testing single classification:\n");
    
    for text in &test_texts {
        println!("Text: \"{}\"", text);
        
        let start = std::time::Instant::now();
        match model.classify(text) {
            Ok(result) => {
                let elapsed = start.elapsed();
                println!("  ✅ Category: {} (class {})", result.category, result.class);
                println!("  ✅ Confidence: {:.2}%", result.confidence * 100.0);
                println!("  ⏱️  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
                
                // Show top 3 probabilities
                let mut indexed_probs: Vec<(usize, f32)> = result.probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                println!("  📊 Top 3:");
                if let Some(cats) = model.categories() {
                    for (i, prob) in indexed_probs.iter().take(3) {
                        println!("     {}: {:.1}%", cats[*i], prob * 100.0);
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Error: {:?}", e);
            }
        }
        println!();
    }
    
    // Test batch classification
    println!("🚀 Testing batch classification:\n");
    
    let start = std::time::Instant::now();
    match model.classify_batch(&test_texts) {
        Ok(results) => {
            let elapsed = start.elapsed();
            
            println!("✅ Batch classification completed!");
            println!("⏱️  Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
            println!("⏱️  Per sample: {:.2}ms", elapsed.as_secs_f64() * 1000.0 / test_texts.len() as f64);
            println!("📈 Throughput: {:.0} samples/sec\n", test_texts.len() as f64 / elapsed.as_secs_f64());
            
            for (text, result) in test_texts.iter().zip(results.iter()) {
                println!("\"{}\"", text);
                println!("  → {} ({:.1}%)", result.category, result.confidence * 100.0);
            }
        }
        Err(e) => {
            println!("❌ Batch error: {:?}", e);
        }
    }
    
    println!("\n✨ Test complete!");
    
    Ok(())
}


