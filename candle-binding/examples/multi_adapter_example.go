// +build example

package main

import (
	"fmt"
	"log"

	candle "github.com/your-org/semantic-router/candle-binding"
)

func main() {
	fmt.Println("=======================================================")
	fmt.Println("   Qwen3 Multi-LoRA Adapter Classification Example")
	fmt.Println("=======================================================")
	fmt.Println()

	// 1. Initialize the multi-adapter classifier with base model
	fmt.Println("📥 Initializing Qwen3 Multi-LoRA Classifier...")
	err := candle.InitQwen3MultiLoRAClassifier("../models/Qwen3-0.6B")
	if err != nil {
		log.Fatalf("Failed to initialize classifier: %v", err)
	}
	fmt.Println("✅ Base model loaded successfully")
	fmt.Println()

	// 2. Load category classification adapter
	fmt.Println("📦 Loading category adapter...")
	err = candle.LoadQwen3LoRAAdapter("category", "../models/qwen3_generative_classifier_r16_fixed")
	if err != nil {
		log.Fatalf("Failed to load category adapter: %v", err)
	}
	fmt.Println("✅ Category adapter loaded")
	fmt.Println()

	// 3. (Optional) Load additional adapters
	// err = candle.LoadQwen3LoRAAdapter("jailbreak", "../models/jailbreak_adapter")
	// err = candle.LoadQwen3LoRAAdapter("sentiment", "../models/sentiment_adapter")

	// 4. List all loaded adapters
	adapters, err := candle.GetQwen3LoadedAdapters()
	if err != nil {
		log.Fatalf("Failed to get loaded adapters: %v", err)
	}
	fmt.Printf("📋 Loaded adapters: %v\n", adapters)
	fmt.Println()

	// 5. Test classification with various texts
	testCases := []struct {
		name string
		text string
	}{
		{
			name: "Economics",
			text: "What is GDP and how does it measure economic growth?",
		},
		{
			name: "Computer Science",
			text: "What is the difference between TCP and UDP protocols?",
		},
		{
			name: "Physics",
			text: "What is Newton's second law of motion?",
		},
		{
			name: "Biology",
			text: "What is the primary function of ribosomes in cells?",
		},
		{
			name: "Law",
			text: "What does 'habeas corpus' mean in legal terms?",
		},
	}

	fmt.Println("🚀 Running Classifications:")
	fmt.Println("-----------------------------------------------------------")

	for i, tc := range testCases {
		result, err := candle.ClassifyWithQwen3Adapter(tc.text, "category")
		if err != nil {
			log.Printf("❌ Failed to classify '%s': %v", tc.name, err)
			continue
		}

		fmt.Printf("\n[%d] %s\n", i+1, tc.name)
		fmt.Printf("    Text: %s\n", tc.text)
		fmt.Printf("    Predicted: %s\n", result.CategoryName)
		fmt.Printf("    Confidence: %.2f%%\n", result.Confidence*100)
		fmt.Printf("    Class ID: %d\n", result.ClassID)

		// Show top 3 probabilities
		fmt.Printf("    Top 3 Categories:\n")
		type categoryProb struct {
			index int
			prob  float32
		}
		var probs []categoryProb
		for idx, prob := range result.Probabilities {
			probs = append(probs, categoryProb{idx, prob})
		}
		// Simple bubble sort for top 3
		for i := 0; i < len(probs); i++ {
			for j := i + 1; j < len(probs); j++ {
				if probs[j].prob > probs[i].prob {
					probs[i], probs[j] = probs[j], probs[i]
				}
			}
		}
		for i := 0; i < 3 && i < len(probs); i++ {
			fmt.Printf("      %d. Class %d: %.2f%%\n", i+1, probs[i].index, probs[i].prob*100)
		}
	}

	fmt.Println()
	fmt.Println("-----------------------------------------------------------")
	fmt.Println("✅ All classifications completed successfully!")
	fmt.Println("=======================================================")
}

