use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::Device;

// Note: In benchmark context, we reference the library crate
extern crate candle_semantic_router;
use candle_semantic_router::model_architectures::generative::qwen3_causal::Qwen3CausalLM;

// Note: This benchmark requires a trained model to run
// Set MODEL_PATH environment variable to your model directory

fn benchmark_single_classification(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "./models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    // Try to load model
    let model = match Qwen3CausalLM::from_pretrained(&model_path, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("⚠️  Model not found: {:?}", e);
            eprintln!("Set MODEL_PATH to run benchmarks");
            return;
        }
    };
    
    let test_text = "What is photosynthesis in plants?";
    
    c.bench_function("classify_single", |b| {
        b.iter(|| {
            model.classify(black_box(test_text))
        })
    });
}

fn benchmark_batch_classification(c: &mut Criterion) {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "./models/qwen3_generative_classifier_r16".to_string());
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    let model = match Qwen3CausalLM::from_pretrained(&model_path, &device) {
        Ok(m) => m,
        Err(_) => return,
    };
    
    let test_texts = vec![
        "What is photosynthesis?",
        "Calculate the derivative of x^2",
        "What causes earthquakes?",
        "Explain supply and demand",
        "What is recursion?",
    ];
    
    let mut group = c.benchmark_group("batch_classification");
    
    for batch_size in [1, 5, 10, 50, 100].iter() {
        let texts: Vec<&str> = test_texts.iter()
            .cycle()
            .take(*batch_size)
            .map(|s| s.as_ref())
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &texts,
            |b, texts| {
                b.iter(|| model.classify_batch(black_box(texts)))
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_classification,
    benchmark_batch_classification
);
criterion_main!(benches);

