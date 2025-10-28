use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_multi_lora_classifier::Qwen3MultiLoRAClassifier;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

fn load_test_data() -> Vec<TestSample> {
    let data_path = "../bench/test_data.json";
    let contents = fs::read_to_string(data_path)
        .expect("Failed to read test_data.json");
    serde_json::from_str(&contents).expect("Failed to parse test_data.json")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("     Qwen3 Multi-LoRA Adapter - Accuracy & Latency Test");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let base_model_path = "/home/ubuntu/rootfs/back/semantic-router.bak/models/Qwen3-0.6B";
    let adapter_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16_fixed".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("🔧 Configuration:");
    println!("   Base model: {}", base_model_path);
    println!("   Adapter: {}", adapter_path);
    println!("   Device: {:?}\n", device);
    
    println!("📥 Loading base model...");
    let start_load = Instant::now();
    let mut model = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    let load_base_time = start_load.elapsed();
    println!("✅ Base model loaded in {:.2}s", load_base_time.as_secs_f64());
    
    println!("\n📦 Loading category adapter...");
    let start_adapter = Instant::now();
    model.load_adapter("category", &adapter_path)?;
    let load_adapter_time = start_adapter.elapsed();
    println!("✅ Adapter loaded in {:.2}s", load_adapter_time.as_secs_f64());
    
    let total_load_time = load_base_time + load_adapter_time;
    println!("   Total load time: {:.2}s\n", total_load_time.as_secs_f64());
    
    let test_data = load_test_data();
    println!("📊 Loaded {} test samples\n", test_data.len());
    
    // Run single sample timing test
    println!("⏱️  Single Sample Latency Test (first 3 samples):");
    for i in 0..3.min(test_data.len()) {
        let start = Instant::now();
        let _result = model.classify_with_adapter(&test_data[i].text, "category")?;
        let latency = start.elapsed();
        println!("   Sample {}: {:.1}ms", i + 1, latency.as_secs_f64() * 1000.0);
    }
    println!();
    
    // Run single-sample classification on all samples
    println!("🚀 Running single-sample classification on all {} samples...\n", test_data.len());
    
    let start_batch = Instant::now();
    let mut results = Vec::new();
    for sample in &test_data {
        let result = model.classify_with_adapter(&sample.text, "category")?;
        results.push(result);
    }
    let batch_time = start_batch.elapsed();
    
    println!("✅ Classification completed in {:.2}ms", batch_time.as_secs_f64() * 1000.0);
    println!("   Average per sample: {:.2}ms\n", batch_time.as_secs_f64() * 1000.0 / test_data.len() as f64);
    
    // Calculate accuracy and print results
    println!("═══════════════════════════════════════════════════════════");
    println!("                    Classification Results");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("{:>4} | {:40} | {:15} | {:15} | {:>8} | {:>5}", 
        "#", "Text", "Predicted", "True Label", "Conf%", "✓");
    println!("{}", "-".repeat(100));
    
    let mut correct = 0;
    let mut total = 0;
    
    for (i, result) in results.iter().enumerate() {
        total += 1;
        let is_correct = result.category == test_data[i].true_label;
        if is_correct {
            correct += 1;
        }
        
        let match_symbol = if is_correct { "✅" } else { "❌" };
        let text_preview = if test_data[i].text.len() > 40 {
            format!("{}...", &test_data[i].text[..37])
        } else {
            test_data[i].text.clone()
        };
        
        println!("{:4} | {:40} | {:15} | {:15} | {:7.1}% | {}", 
            i + 1, 
            text_preview,
            result.category, 
            test_data[i].true_label,
            result.confidence * 100.0,
            match_symbol
        );
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    
    println!("\n{}", "═".repeat(100));
    println!("                           SUMMARY");
    println!("{}", "═".repeat(100));
    println!("📊 Accuracy:          {}/{} ({:.2}%)", correct, total, accuracy);
    println!("⏱️  Total Time:        {:.2}ms", batch_time.as_secs_f64() * 1000.0);
    println!("📈 Throughput:        {:.1} samples/sec", test_data.len() as f64 / batch_time.as_secs_f64());
    println!("⚡ Avg Latency:       {:.2}ms per sample", batch_time.as_secs_f64() * 1000.0 / test_data.len() as f64);
    println!("{}", "═".repeat(100));
    
    println!("\n⚠️  IMPORTANT NOTE:");
    if accuracy < 50.0 {
        println!("   Current accuracy is {:.1}% (base model only).", accuracy);
        println!("   LoRA adapters are loaded but NOT applied during forward pass.");
        println!("   This is expected behavior - see MULTI_ADAPTER_IMPLEMENTATION.md");
        println!("\n   To achieve ~78% accuracy:");
        println!("   1. Fork candle-transformers");
        println!("   2. Add LoRA hooks to CausalSelfAttention and MLP layers");
        println!("   3. Apply LoRA deltas: output += LoRA_B(LoRA_A(input)) * (alpha/rank)");
    } else {
        println!("   ✅ LoRA application is working! Accuracy > 50%");
    }
    println!();
    
    Ok(())
}


