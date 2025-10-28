use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::Device;
use std::fs;
use serde::{Deserialize, Serialize};

extern crate candle_semantic_router;
use candle_semantic_router::model_architectures::generative::qwen3_causal::Qwen3CausalLM;

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

fn benchmark_single_classification_real_data(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/Qwen3-0.6B".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("\n🔧 Loading model from: {}", model_path);
    println!("🖥️  Using device: {:?}", device);
    
    let model = match Qwen3CausalLM::from_pretrained(&model_path, &device) {
        Ok(m) => {
            println!("✅ Model loaded successfully");
            if let Some(cats) = m.categories() {
                println!("📚 Categories: {}", cats.join(", "));
            }
            m
        }
        Err(e) => {
            eprintln!("❌ Failed to load model: {:?}", e);
            eprintln!("\n💡 Set MODEL_PATH to a valid Qwen3 model directory");
            eprintln!("   Example: MODEL_PATH=../models/Qwen3-0.6B cargo bench");
            return;
        }
    };
    
    let test_data = load_test_data();
    println!("\n📊 Loaded {} test samples", test_data.len());
    
    // Benchmark first 10 samples individually
    let mut group = c.benchmark_group("single_classification");
    
    for (i, sample) in test_data.iter().take(10).enumerate() {
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

fn benchmark_batch_classification_real_data(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/Qwen3-0.6B".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let model = match Qwen3CausalLM::from_pretrained(&model_path, &device) {
        Ok(m) => m,
        Err(_) => return,
    };
    
    let test_data = load_test_data();
    let texts: Vec<&str> = test_data.iter().map(|s| s.text.as_str()).collect();
    
    let mut group = c.benchmark_group("batch_classification");
    group.sample_size(10); // Reduce samples for faster benchmarking
    
    // Benchmark different batch sizes
    for &batch_size in &[1, 5, 10, 20, 50, 67] {
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

fn benchmark_throughput(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "../models/Qwen3-0.6B".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let model = match Qwen3CausalLM::from_pretrained(&model_path, &device) {
        Ok(m) => m,
        Err(_) => return,
    };
    
    let test_data = load_test_data();
    let all_texts: Vec<&str> = test_data.iter().map(|s| s.text.as_str()).collect();
    
    c.bench_function("throughput_all_67_samples", |b| {
        b.iter(|| model.classify_batch(black_box(&all_texts)))
    });
}

criterion_group!(
    benches,
    benchmark_single_classification_real_data,
    benchmark_batch_classification_real_data,
    benchmark_throughput
);
criterion_main!(benches);


