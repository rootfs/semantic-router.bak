use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_causal::Qwen3CausalLM;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("   Qwen3 BASE Model - Quick Accuracy Test (QK-norm Fix)");
    println!("═══════════════════════════════════════════════════════════\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/Qwen3-0.6B".to_string());
    
    let device = Device::cuda_if_available(0).expect("Failed to get device");
    
    println!("📥 Loading model from: {}", model_path);
    println!("🖥️  Device: {:?}\n", device);
    
    let model = Qwen3CausalLM::from_pretrained(&model_path, &device)
        .expect("Failed to load model");
    
    // Load test data
    let test_path = "../bench/test_data.json";
    let test_data: Vec<TestSample> = serde_json::from_str(
        &fs::read_to_string(test_path).expect("Failed to read test data")
    ).expect("Failed to parse test data");
    
    println!("🧪 Testing on {} samples...\n", test_data.len());
    
    let mut correct = 0;
    let mut total_time = std::time::Duration::ZERO;
    
    for (i, sample) in test_data.iter().enumerate().take(20) {
        let start = std::time::Instant::now();
        let result = model.classify(&sample.text).expect("Classification failed");
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        let is_correct = result.category == sample.true_label;
        if is_correct {
            correct += 1;
        }
        
        println!("Sample {}: {}", i + 1, if is_correct { "✅" } else { "❌" });
        println!("  Text: {}...", &sample.text.chars().take(60).collect::<String>());
        println!("  True: {} | Predicted: {} ({:.1}%)", 
            sample.true_label, result.category, result.confidence * 100.0);
        println!("  Time: {:.2}ms\n", elapsed.as_secs_f64() * 1000.0);
    }
    
    println!("═══════════════════════════════════════════════════════════");
    println!("📊 Results:");
    println!("   Accuracy: {}/{} ({:.1}%)", correct, 20, correct as f64 / 20.0 * 100.0);
    println!("   Avg Time: {:.2}ms", total_time.as_secs_f64() * 1000.0 / 20.0);
    println!("═══════════════════════════════════════════════════════════");
}


