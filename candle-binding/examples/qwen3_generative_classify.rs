//! Example: Using Qwen3 for Fast Classification with Logit Extraction
//!
//! This example demonstrates the logit extraction approach which:
//! - Uses a single forward pass (fast!)
//! - Returns probability distributions
//! - Supports batch processing
//!
//! Run with:
//! ```bash
//! cargo run --example qwen3_generative_classify --features cuda -- \
//!     --model-path ./models/qwen3_generative_classifier_r16 \
//!     --text "What is photosynthesis?"
//! 
//! # Batch mode
//! cargo run --example qwen3_generative_classify --features cuda -- \
//!     --model-path ./models/qwen3_generative_classifier_r16 \
//!     --batch
//! ```

use anyhow::Result;
use candle_core::Device;

// Import from semantic_router if available, else define simple args
#[cfg(feature = "clap")]
use clap::Parser;

use semantic_router::model_architectures::generative::qwen3_causal::Qwen3CausalLM;

#[cfg_attr(feature = "clap", derive(Parser))]
#[cfg_attr(feature = "clap", command(author, version, about, long_about = None))]
#[derive(Debug)]
struct Args {
    /// Path to the model directory
    #[arg(long)]
    model_path: String,

    /// Text to classify (single mode)
    #[arg(long)]
    text: Option<String>,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Run batch classification demo
    #[arg(long)]
    batch: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    println!("Loading model from: {}", args.model_path);
    println!("Device: {:?}", device);

    // Load model
    let model = Qwen3CausalLM::from_pretrained(&args.model_path, &device)?;

    // Show available categories
    if let Some(categories) = model.categories() {
        println!("\n📚 Available categories ({}):", categories.len());
        for (i, cat) in categories.iter().enumerate() {
            println!("  [{}] {}", i, cat);
        }
    }

    if args.batch {
        // Batch classification demo
        println!("\n🚀 Running batch classification demo...");
        println!("{}", "=".repeat(60));
        
        let test_texts = vec![
            "What is photosynthesis?",
            "How do I calculate compound interest?",
            "What causes earthquakes?",
            "Explain supply and demand.",
            "What is recursion in programming?",
        ];
        
        let start = std::time::Instant::now();
        let results = model.classify_batch(&test_texts)?;
        let elapsed = start.elapsed();
        
        println!("\n📊 Batch Results:");
        for (text, result) in test_texts.iter().zip(results.iter()) {
            println!("\nText: \"{}\"", text);
            println!("  → Category: {} (class {})", result.category, result.class);
            println!("  → Confidence: {:.2}%", result.confidence * 100.0);
            println!("  → Top 3 probabilities:");
            
            let mut indexed_probs: Vec<(usize, f32)> = result.probabilities
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            for (i, prob) in indexed_probs.iter().take(3) {
                if let Some(cats) = model.categories() {
                    println!("     {}: {:.2}%", cats[*i], prob * 100.0);
                }
            }
        }
        
        println!("\n⚡ Performance:");
        println!("  Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Per sample: {:.2}ms", elapsed.as_secs_f64() * 1000.0 / test_texts.len() as f64);
        println!("  Throughput: {:.0} samples/sec", test_texts.len() as f64 / elapsed.as_secs_f64());
        
    } else if let Some(text) = args.text {
        // Single classification
        println!("\n🔍 Classifying: \"{}\"", text);
        println!("{}", "=".repeat(60));

        let start = std::time::Instant::now();
        let result = model.classify(&text)?;
        let elapsed = start.elapsed();

        println!("\n📋 Classification Result:");
        println!("  Category: {} (class {})", result.category, result.class);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        
        println!("\n📊 Probability Distribution:");
        let mut indexed_probs: Vec<(usize, f32)> = result.probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (i, prob) in indexed_probs.iter() {
            if let Some(cats) = model.categories() {
                let bar_len = (prob * 50.0) as usize;
                let bar = "█".repeat(bar_len);
                println!("  {:20} {:>6.2}% {}", cats[*i], prob * 100.0, bar);
            }
        }
        
        println!("\n⚡ Performance:");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        
    } else {
        println!("\nError: Please provide --text \"your text\" or use --batch flag");
        std::process::exit(1);
    }

    Ok(())
}

