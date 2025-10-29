use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::Device;
use std::fs;
use serde::{Deserialize, Serialize};

extern crate candle_semantic_router;
use candle_semantic_router::model_architectures::generative::qwen3_lora_classifier::Qwen3LoRAClassifier;

#[derive(Debug, Deserialize, Serialize)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

fn load_test_data() -> Vec<TestSample> {
    let data_path = "/home/ubuntu/rootfs/back/semantic-router.bak/bench/results/json/test_data.json";
    let contents = fs::read_to_string(data_path)
        .expect("Failed to read test_data.json");
    serde_json::from_str(&contents).expect("Failed to parse test_data.json")
}

fn benchmark_lora_single_classification(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("\n🔧 Loading LoRA model from: {}", model_path);
    println!("🖥️  Using device: {:?}", device);
    
    let model = match Qwen3LoRAClassifier::from_pretrained(&model_path, &device) {
        Ok(m) => {
            println!("✅ LoRA model loaded successfully");
            println!("📚 Categories: {:?}", m.categories());
            m
        }
        Err(e) => {
            eprintln!("❌ Failed to load LoRA model: {:?}", e);
            eprintln!("\n💡 Set MODEL_PATH to your LoRA adapter directory");
            eprintln!("   Example: MODEL_PATH=../models/qwen3_generative_classifier_r16 cargo bench --bench lora_benchmark");
            return;
        }
    };
    
    let test_data = load_test_data();
    println!("\n📊 Loaded {} test samples", test_data.len());
    
    // Benchmark first 3 samples individually with reduced iterations
    let mut group = c.benchmark_group("lora_single_classification");
    group.sample_size(10);  // Only 10 samples per benchmark
    group.measurement_time(std::time::Duration::from_secs(10));  // 10 second measurement
    
    for (i, sample) in test_data.iter().take(3).enumerate() {
        group.bench_with_input(
            BenchmarkId::new("sample", i),
            &sample.text,
            |b, text| {
                b.iter(|| model.classify(black_box(text)))
            }
        );
    }
    
    group.finish();
}

fn benchmark_lora_batch_classification(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let model = match Qwen3LoRAClassifier::from_pretrained(&model_path, &device) {
        Ok(m) => m,
        Err(_) => return,
    };
    
    let test_data = load_test_data();
    let texts: Vec<&str> = test_data.iter().map(|s| s.text.as_str()).collect();
    
    let mut group = c.benchmark_group("lora_batch_classification");
    group.sample_size(10); // Reduce samples for faster benchmarking
    group.measurement_time(std::time::Duration::from_secs(10));  // 10 second measurement
    
    // Benchmark different batch sizes (reduced set)
    for &batch_size in &[1, 5, 10, 20] {
        if batch_size > texts.len() {
            continue;
        }
        
        let batch = &texts[..batch_size];
        
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            batch,
            |b, batch| {
                b.iter(|| model.classify_batch(black_box(batch)))
            }
        );
    }
    
    group.finish();
}

fn benchmark_lora_accuracy(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let model = match Qwen3LoRAClassifier::from_pretrained(&model_path, &device) {
        Ok(m) => m,
        Err(_) => return,
    };
    
    let test_data = load_test_data();
    let all_texts: Vec<&str> = test_data.iter().map(|s| s.text.as_str()).collect();
    
    // Run classification once to compute accuracy
    println!("\n=== Accuracy Evaluation ===");
    println!("Running classification on {} samples...\n", all_texts.len());
    
    match model.classify_batch(&all_texts) {
        Ok(results) => {
            let mut correct = 0;
            let mut total = 0;
            
            println!("Sample | Text | Predicted | True | Confidence | Match");
            println!("{}", "-".repeat(80));
            
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
                
                println!("{:4} | {:40} | {:15} | {:15} | {:5.1}% | {}", 
                    i + 1, 
                    text_preview,
                    result.category, 
                    test_data[i].true_label,
                    result.confidence * 100.0,
                    match_symbol
                );
            }
            
            let accuracy = (correct as f64 / total as f64) * 100.0;
            println!("\n{}", "=".repeat(80));
            println!("📊 Final Accuracy: {}/{} ({:.2}%)", correct, total, accuracy);
            println!("{}\n", "=".repeat(80));
        }
        Err(e) => {
            eprintln!("Failed to classify batch: {:?}", e);
        }
    }
    
    let mut group = c.benchmark_group("lora_throughput");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));
    
    group.bench_function("all_samples", |b| {
        b.iter(|| model.classify_batch(black_box(&all_texts)))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_lora_single_classification,
    benchmark_lora_batch_classification,
    benchmark_lora_accuracy
);
criterion_main!(benches);

