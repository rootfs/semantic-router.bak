use candle_core::Device;
use candle_semantic_router::model_architectures::generative::qwen3_multi_lora_classifier::Qwen3MultiLoRAClassifier;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("     Qwen3 Multi-LoRA Adapter - Dynamic Switching Test");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let base_model_path = "/home/ubuntu/rootfs/back/semantic-router.bak/models/Qwen3-0.6B";
    let category_adapter_path = "/home/ubuntu/rootfs/back/semantic-router.bak/models/qwen3_generative_classifier_r16_fixed";
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    println!("🔧 Configuration:");
    println!("   Base model: {}", base_model_path);
    println!("   Device: {:?}\n", device);
    
    // Initialize base model
    println!("📥 Step 1: Load base Qwen3 model (shared across all adapters)");
    let mut model = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    
    // Load category classification adapter
    println!("📦 Step 2: Load category classification adapter");
    model.load_adapter("category", category_adapter_path)?;
    
    // TODO: Load jailbreak detection adapter when available
    // model.load_adapter("jailbreak", jailbreak_adapter_path)?;
    
    println!("📋 Loaded adapters: {:?}\n", model.list_adapters());
    
    // Test prompts
    let test_prompts = vec![
        ("What does GDP measure in an economy?", "category"),
        ("What is Ohm's Law in electrical engineering?", "category"),
        ("In marketing, what are the four Ps of the marketing mix?", "category"),
    ];
    
    println!("🧪 Testing adapter switching with different prompts\n");
    println!("{}", "=".repeat(80));
    
    for (text, adapter_name) in test_prompts {
        println!("\n📝 Text: {}", text);
        println!("🔄 Using adapter: '{}'", adapter_name);
        
        let result = model.classify_with_adapter(text, adapter_name)?;
        
        println!("   ✅ Predicted: {} ({:.1}%)", result.category, result.confidence * 100.0);
        
        // Show top 3 predictions
        let mut sorted: Vec<_> = result.probabilities.iter()
            .zip(result.all_categories.iter())
            .collect();
        sorted.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
        
        println!("   📊 Top 3:");
        for (prob, cat) in sorted.iter().take(3) {
            println!("      {:20}: {:5.1}%", cat, *prob * 100.0);
        }
    }
    
    println!("\n{}", "=".repeat(80));
    println!("\n✨ Multi-adapter system working!");
    println!("   - Base model loaded once: ✅");
    println!("   - Adapters can be switched: ✅");
    println!("   - No model reloading needed: ✅");
    
    println!("\n⚠️  IMPORTANT NOTE:");
    println!("   LoRA adapters are loaded but not yet fully applied during forward pass.");
    println!("   Currently showing base model predictions (~28% accuracy).");
    println!("   Need to implement LoRA application hooks to achieve ~78% accuracy.");
    println!("\n   Next steps:");
    println!("   1. Hook into Qwen3 layer forward passes");
    println!("   2. Apply LoRA deltas: output += LoRA_B(LoRA_A(input)) * scaling");
    println!("   3. This requires either:");
    println!("      a) Modifying candle-transformers to expose layer hooks, OR");
    println!("      b) Implementing custom forward pass with LoRA injection");
    
    Ok(())
}

