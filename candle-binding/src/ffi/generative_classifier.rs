//! FFI bindings for Qwen3 LoRA Generative Classifier
//!
//! Exposes the Qwen3 + LoRA classifier to Go via C ABI.

use crate::model_architectures::generative::Qwen3LoRAClassifier;
use candle_core::Device;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::OnceLock;

/// Global classifier instance
static GLOBAL_QWEN3_CLASSIFIER: OnceLock<Qwen3LoRAClassifier> = OnceLock::new();

/// Generative classification result returned to Go
#[repr(C)]
pub struct GenerativeClassificationResult {
    /// Predicted class index
    pub class_id: i32,
    
    /// Confidence score (probability)
    pub confidence: f32,
    
    /// Category name (null-terminated C string, must be freed by caller)
    pub category_name: *mut c_char,
    
    /// Probabilities for all categories (array, must be freed by caller)
    pub probabilities: *mut f32,
    
    /// Number of categories
    pub num_categories: i32,
    
    /// Error flag (true if error occurred)
    pub error: bool,
    
    /// Error message (null-terminated C string, only set if error=true, must be freed by caller)
    pub error_message: *mut c_char,
}

impl Default for GenerativeClassificationResult {
    fn default() -> Self {
        Self {
            class_id: -1,
            confidence: 0.0,
            category_name: ptr::null_mut(),
            probabilities: ptr::null_mut(),
            num_categories: 0,
            error: true,
            error_message: ptr::null_mut(),
        }
    }
}

/// Initialize Qwen3 LoRA classifier
///
/// # Arguments
/// - `model_path`: Path to model directory containing config.json, model.safetensors, tokenizer.json, label_mapping.json
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_qwen3_lora_classifier(model_path: *const c_char) -> i32 {
    if model_path.is_null() {
        eprintln!("Error: model_path is null");
        return -1;
    }

    let model_path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_path: {}", e);
                return -1;
            }
        }
    };

    // Determine device (use CPU for now, can add GPU support later)
    let device = Device::Cpu;

    // Check if already initialized
    if GLOBAL_QWEN3_CLASSIFIER.get().is_some() {
        println!("✅ Qwen3 LoRA classifier already initialized, reusing existing instance");
        return 0;
    }

    // Load classifier
    match Qwen3LoRAClassifier::from_pretrained(model_path_str, &device) {
        Ok(classifier) => {
            match GLOBAL_QWEN3_CLASSIFIER.set(classifier) {
                Ok(_) => {
                    println!("✅ Qwen3 LoRA classifier initialized from: {}", model_path_str);
                    0
                }
                Err(_) => {
                    // This shouldn't happen since we checked above, but handle it
                    println!("✅ Qwen3 LoRA classifier already initialized (race condition), reusing");
                    0
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to load Qwen3 LoRA classifier: {}", e);
            -1
        }
    }
}

/// Classify text using Qwen3 LoRA classifier
///
/// # Arguments
/// - `text`: Input text to classify (null-terminated C string)
/// - `result`: Pointer to GenerativeClassificationResult struct (allocated by caller)
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - `result` must be a valid pointer to GenerativeClassificationResult
/// - Caller must free: result.category_name, result.probabilities, result.error_message
#[no_mangle]
pub extern "C" fn classify_text_qwen3_lora(
    text: *const c_char,
    result: *mut GenerativeClassificationResult,
) -> i32 {
    if text.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to classify_text_qwen3_lora");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    // Get classifier
    let classifier = match GLOBAL_QWEN3_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 LoRA classifier not initialized");
            unsafe {
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message("Classifier not initialized");
            }
            return -1;
        }
    };

    // Classify
    match classifier.classify(text_str) {
        Ok(classification_result) => {
            // Extract fields from ClassificationResult
            let best_idx = classification_result.class as usize;
            let best_category = classification_result.category;
            let best_confidence = classification_result.confidence;
            let mut probabilities = classification_result.probabilities;
            
            // Allocate C strings and arrays
            let category_name_c = match CString::new(best_category.as_str()) {
                Ok(s) => s.into_raw(),
                Err(e) => {
                    eprintln!("Error: failed to create category name C string: {}", e);
                    unsafe {
                        (*result) = GenerativeClassificationResult::default();
                        (*result).error_message = create_error_message(&format!("Failed to create C string: {}", e));
                    }
                    return -1;
                }
            };
            
            // Allocate probabilities array
            let probs_ptr = probabilities.as_mut_ptr();
            let num_categories = probabilities.len();
            std::mem::forget(probabilities); // Prevent Rust from deallocating
            
            unsafe {
                (*result) = GenerativeClassificationResult {
                    class_id: best_idx as i32,
                    confidence: best_confidence,
                    category_name: category_name_c,
                    probabilities: probs_ptr,
                    num_categories: num_categories as i32,
                    error: false,
                    error_message: ptr::null_mut(),
                };
            }
            
            0
        }
        Err(e) => {
            eprintln!("Error: classification failed: {}", e);
            unsafe {
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message(&format!("Classification failed: {}", e));
            }
            -1
        }
    }
}

/// Get list of categories from the classifier
///
/// # Arguments
/// - `categories_out`: Output pointer that will be set to point to array of C strings
/// - `num_categories`: Output parameter for number of categories
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - Caller must free each string in the categories array and the array itself
#[no_mangle]
pub extern "C" fn get_qwen3_lora_categories(
    categories_out: *mut *mut *mut c_char,
    num_categories: *mut i32,
) -> i32 {
    if categories_out.is_null() || num_categories.is_null() {
        eprintln!("Error: null pointer passed to get_qwen3_lora_categories");
        return -1;
    }

    let classifier = match GLOBAL_QWEN3_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 LoRA classifier not initialized");
            return -1;
        }
    };

    let cats = classifier.categories();
    let count = cats.len();

    // Allocate array of C strings
    let mut c_strings: Vec<*mut c_char> = Vec::with_capacity(count);
    for cat in cats {
        match CString::new(cat.as_str()) {
            Ok(s) => c_strings.push(s.into_raw()),
            Err(e) => {
                eprintln!("Error: failed to create category C string: {}", e);
                // Free already allocated strings
                for ptr in c_strings {
                    unsafe { let _ = CString::from_raw(ptr); }
                }
                return -1;
            }
        }
    }

    // Transfer ownership to caller
    unsafe {
        *num_categories = count as i32;
        *categories_out = c_strings.as_mut_ptr();
    }
    std::mem::forget(c_strings);

    0
}

/// Free classification result
///
/// # Safety
/// - Must only be called once per result
/// - Result must have been allocated by classify_text_qwen3_lora
#[no_mangle]
pub extern "C" fn free_generative_classification_result(result: *mut GenerativeClassificationResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        // Free category name
        if !(*result).category_name.is_null() {
            let _ = CString::from_raw((*result).category_name);
        }

        // Free probabilities array
        if !(*result).probabilities.is_null() {
            let num_cats = (*result).num_categories as usize;
            let _ = Vec::from_raw_parts((*result).probabilities, num_cats, num_cats);
        }

        // Free error message
        if !(*result).error_message.is_null() {
            let _ = CString::from_raw((*result).error_message);
        }
    }
}

/// Free categories array
///
/// # Safety
/// - Must only be called once per array
/// - Array must have been allocated by get_qwen3_lora_categories
#[no_mangle]
pub extern "C" fn free_categories(categories: *mut *mut c_char, num_categories: i32) {
    if categories.is_null() || num_categories <= 0 {
        return;
    }

    unsafe {
        for i in 0..num_categories {
            let ptr = *categories.offset(i as isize);
            if !ptr.is_null() {
                let _ = CString::from_raw(ptr);
            }
        }
        let _ = Vec::from_raw_parts(categories, num_categories as usize, num_categories as usize);
    }
}

/// Helper: create error message C string
fn create_error_message(msg: &str) -> *mut c_char {
    match CString::new(msg) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_via_classify() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

