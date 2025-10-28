// Test Go bindings with Qwen3 Multi-LoRA adapter using benchmark dataset
// Run with: go run test_go_multi_adapter.go

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    int class_id;
    float confidence;
    char* category_name;
    float* probabilities;
    int num_categories;
    bool error;
    char* error_message;
} GenerativeClassificationResult;

extern int init_qwen3_multi_lora_classifier(const char* base_model_path);
extern int load_qwen3_lora_adapter(const char* adapter_name, const char* adapter_path);
extern int classify_with_qwen3_adapter(const char* text, const char* adapter_name, GenerativeClassificationResult* result);
extern int get_qwen3_loaded_adapters(char*** adapters_out, int* num_adapters);
extern void free_generative_classification_result(GenerativeClassificationResult* result);
extern void free_categories(char** categories, int num_categories);
*/
import "C"
import "unsafe"

// TestSample represents a single test case from test_data.json
type TestSample struct {
	Text         string `json:"text"`
	TrueLabel    string `json:"true_label"`
	TrueLabelID  int    `json:"true_label_id"`
}

// Qwen3LoRAResult represents classification result
type Qwen3LoRAResult struct {
	ClassID       int
	Confidence    float32
	CategoryName  string
	Probabilities []float32
	NumCategories int
}

// InitQwen3MultiLoRAClassifier initializes the classifier with base model
func InitQwen3MultiLoRAClassifier(baseModelPath string) error {
	cBaseModelPath := C.CString(baseModelPath)
	defer C.free(unsafe.Pointer(cBaseModelPath))

	result := C.init_qwen3_multi_lora_classifier(cBaseModelPath)
	if result != 0 {
		return fmt.Errorf("failed to initialize Qwen3 Multi-LoRA classifier (error code: %d)", result)
	}

	log.Printf("✅ Qwen3 Multi-LoRA classifier initialized from: %s", baseModelPath)
	return nil
}

// LoadQwen3LoRAAdapter loads a LoRA adapter
func LoadQwen3LoRAAdapter(adapterName, adapterPath string) error {
	cAdapterName := C.CString(adapterName)
	defer C.free(unsafe.Pointer(cAdapterName))

	cAdapterPath := C.CString(adapterPath)
	defer C.free(unsafe.Pointer(cAdapterPath))

	result := C.load_qwen3_lora_adapter(cAdapterName, cAdapterPath)
	if result != 0 {
		return fmt.Errorf("failed to load adapter '%s' (error code: %d)", adapterName, result)
	}

	log.Printf("✅ Loaded adapter '%s' from: %s", adapterName, adapterPath)
	return nil
}

// ClassifyWithQwen3Adapter classifies text using a specific adapter
func ClassifyWithQwen3Adapter(text, adapterName string) (*Qwen3LoRAResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cAdapterName := C.CString(adapterName)
	defer C.free(unsafe.Pointer(cAdapterName))

	var result C.GenerativeClassificationResult
	ret := C.classify_with_qwen3_adapter(cText, cAdapterName, &result)
	defer C.free_generative_classification_result(&result)

	if ret != 0 || result.error {
		errMsg := fmt.Sprintf("classification with adapter '%s' failed", adapterName)
		if result.error_message != nil {
			errMsg = C.GoString(result.error_message)
		}
		return nil, fmt.Errorf("%s", errMsg)
	}

	// Convert probabilities
	numCats := int(result.num_categories)
	probs := make([]float32, numCats)
	if result.probabilities != nil && numCats > 0 {
		probsSlice := (*[1000]C.float)(unsafe.Pointer(result.probabilities))[:numCats:numCats]
		for i := 0; i < numCats; i++ {
			probs[i] = float32(probsSlice[i])
		}
	}

	goResult := &Qwen3LoRAResult{
		ClassID:       int(result.class_id),
		Confidence:    float32(result.confidence),
		CategoryName:  C.GoString(result.category_name),
		Probabilities: probs,
		NumCategories: numCats,
	}

	return goResult, nil
}

func main() {
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  Go Bindings Test - Qwen3 Multi-LoRA Accuracy Test")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Configuration
	baseModelPath := os.Getenv("MODEL_PATH")
	if baseModelPath == "" {
		baseModelPath = "../models/Qwen3-0.6B"
	}
	
	adapterPath := os.Getenv("ADAPTER_PATH")
	if adapterPath == "" {
		adapterPath = "../models/qwen3_generative_classifier_r16_fixed"
	}
	
	testDataPath := "../bench/test_data.json"

	fmt.Printf("🔧 Configuration:\n")
	fmt.Printf("   Base model: %s\n", baseModelPath)
	fmt.Printf("   Adapter: %s\n", adapterPath)
	fmt.Printf("   Test data: %s\n", testDataPath)
	fmt.Println()

	// 1. Initialize multi-LoRA classifier
	fmt.Println("📥 Loading base model...")
	startLoad := time.Now()
	err := InitQwen3MultiLoRAClassifier(baseModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize classifier: %v", err)
	}
	loadTime := time.Since(startLoad)
	fmt.Printf("✅ Base model loaded in %.2fs\n\n", loadTime.Seconds())

	// 2. Load category adapter
	fmt.Println("📦 Loading category adapter...")
	startAdapter := time.Now()
	err = LoadQwen3LoRAAdapter("category", adapterPath)
	if err != nil {
		log.Fatalf("Failed to load adapter: %v", err)
	}
	adapterTime := time.Since(startAdapter)
	fmt.Printf("✅ Adapter loaded in %.2fs\n", adapterTime.Seconds())
	fmt.Printf("   Total load time: %.2fs\n\n", (loadTime + adapterTime).Seconds())

	// 3. Load test data
	fmt.Println("📊 Loading test data...")
	data, err := ioutil.ReadFile(testDataPath)
	if err != nil {
		log.Fatalf("Failed to read test data: %v", err)
	}

	var testSamples []TestSample
	err = json.Unmarshal(data, &testSamples)
	if err != nil {
		log.Fatalf("Failed to parse test data: %v", err)
	}
	fmt.Printf("✅ Loaded %d test samples\n\n", len(testSamples))

	// 4. Warmup (first call is slower)
	fmt.Println("⏱️  Single Sample Latency Test (first 3 samples):")
	for i := 0; i < 3 && i < len(testSamples); i++ {
		start := time.Now()
		_, err := ClassifyWithQwen3Adapter(testSamples[i].Text, "category")
		elapsed := time.Since(start)
		if err != nil {
			log.Printf("   Sample %d: Error - %v", i+1, err)
		} else {
			fmt.Printf("   Sample %d: %.1fms\n", i+1, elapsed.Seconds()*1000)
		}
	}
	fmt.Println()

	// 5. Run classification on all samples
	fmt.Printf("🚀 Running classification on all %d samples...\n\n", len(testSamples))
	
	startClassification := time.Now()
	correct := 0
	incorrect := 0
	errors := 0
	
	type ResultRecord struct {
		Index        int
		Text         string
		Predicted    string
		TrueLabel    string
		Confidence   float32
		Correct      bool
		Error        error
	}
	
	results := make([]ResultRecord, 0, len(testSamples))
	
	for i, sample := range testSamples {
		result, err := ClassifyWithQwen3Adapter(sample.Text, "category")
		
		record := ResultRecord{
			Index:     i + 1,
			Text:      sample.Text,
			TrueLabel: sample.TrueLabel,
		}
		
		if err != nil {
			record.Error = err
			errors++
		} else {
			record.Predicted = result.CategoryName
			record.Confidence = result.Confidence
			
			// Normalize for comparison
			predicted := strings.ToLower(strings.TrimSpace(result.CategoryName))
			expected := strings.ToLower(strings.TrimSpace(sample.TrueLabel))
			
			if predicted == expected {
				record.Correct = true
				correct++
			} else {
				record.Correct = false
				incorrect++
			}
		}
		
		results = append(results, record)
	}
	
	totalTime := time.Since(startClassification)
	
	fmt.Println("✅ Classification completed")
	fmt.Printf("   Total time: %.2fms\n", totalTime.Seconds()*1000)
	fmt.Printf("   Average per sample: %.2fms\n\n", totalTime.Seconds()*1000/float64(len(testSamples)))

	// 6. Print results
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("                Classification Results")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Printf("%4s | %-40s | %-15s | %-15s | %8s | %5s\n", 
		"#", "Text", "Predicted", "True Label", "Conf%", "✓")
	fmt.Println(strings.Repeat("-", 100))
	
	for _, r := range results {
		textPreview := r.Text
		if len(textPreview) > 40 {
			textPreview = textPreview[:37] + "..."
		}
		
		checkMark := "❌"
		if r.Correct {
			checkMark = "✅"
		}
		if r.Error != nil {
			checkMark = "⚠️"
		}
		
		predicted := r.Predicted
		if r.Error != nil {
			predicted = "ERROR"
		}
		if len(predicted) > 15 {
			predicted = predicted[:12] + "..."
		}
		
		trueLabel := r.TrueLabel
		if len(trueLabel) > 15 {
			trueLabel = trueLabel[:12] + "..."
		}
		
		confPercent := r.Confidence * 100
		
		fmt.Printf("%4d | %-40s | %-15s | %-15s | %7.1f%% | %s\n",
			r.Index, textPreview, predicted, trueLabel, confPercent, checkMark)
	}
	
	// 7. Print summary
	fmt.Println()
	fmt.Println(strings.Repeat("═", 100))
	fmt.Println("                           SUMMARY")
	fmt.Println(strings.Repeat("═", 100))
	
	total := len(testSamples)
	accuracy := float64(correct) / float64(total) * 100
	throughput := float64(total) / totalTime.Seconds()
	avgLatency := totalTime.Seconds() * 1000 / float64(total)
	
	fmt.Printf("📊 Accuracy:          %d/%d (%.2f%%)\n", correct, total, accuracy)
	fmt.Printf("⏱️  Total Time:        %.2fms\n", totalTime.Seconds()*1000)
	fmt.Printf("📈 Throughput:        %.1f samples/sec\n", throughput)
	fmt.Printf("⚡ Avg Latency:       %.2fms per sample\n", avgLatency)
	
	if errors > 0 {
		fmt.Printf("⚠️  Errors:            %d\n", errors)
	}
	
	fmt.Println(strings.Repeat("═", 100))
	fmt.Println()
	
	if accuracy >= 70.0 {
		fmt.Println("✅ EXCELLENT: Go bindings working correctly with LoRA!")
		fmt.Println("   Accuracy matches expected ~71% (Rust implementation)")
	} else if accuracy >= 25.0 {
		fmt.Println("⚠️  WARNING: Accuracy is lower than expected")
		fmt.Println("   Expected: ~71% with LoRA, ~28% without LoRA")
		fmt.Printf("   Got: %.2f%%\n", accuracy)
	} else {
		fmt.Println("❌ ERROR: Accuracy is very low")
		fmt.Println("   This suggests LoRA may not be applied correctly")
	}
	
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
}

