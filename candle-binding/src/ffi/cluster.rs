//! Cluster-based Model Routing FFI Module
//!
//! This module implements K-means clustering for intelligent model routing.
//! It learns model performance patterns from experience data and routes
//! new queries to the optimal model based on semantic similarity.

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::OnceLock;

/// Global singleton for cluster router state
static CLUSTER_ROUTER_STATE: OnceLock<ClusterRouterState> = OnceLock::new();

/// Model performance record for a single query in the experience database
#[derive(Debug, Clone)]
pub struct ExperienceRecord {
    /// Embedding vector for the query
    pub embedding: Vec<f32>,
    /// Model performance scores (model_name -> score)
    /// Score is typically 1.0 for correct, 0.0 for incorrect
    pub model_scores: HashMap<String, f32>,
}

/// Cluster routing state containing trained model
#[derive(Debug)]
pub struct ClusterRouterState {
    /// K-means cluster centers [n_clusters, embedding_dim]
    pub cluster_centers: Tensor,
    /// Best model for each cluster (cluster_id -> model_name)
    pub cluster_best_model: Vec<String>,
    /// Model performance scores per cluster (cluster_id -> model_name -> avg_score)
    pub cluster_model_scores: Vec<HashMap<String, f32>>,
    /// Model cost factors for cost-aware routing (model_name -> relative_cost)
    pub model_costs: HashMap<String, f32>,
    /// List of available model names
    pub model_names: Vec<String>,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Device for tensor operations
    device: Device,
}

/// Result of cluster routing
#[repr(C)]
pub struct ClusterRouteResult {
    /// Best model name (C string, caller must free with free_cstring)
    pub model_name: *mut c_char,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Cluster ID the query was assigned to
    pub cluster_id: i32,
    /// Whether an error occurred
    pub error: bool,
}

impl Default for ClusterRouteResult {
    fn default() -> Self {
        Self {
            model_name: std::ptr::null_mut(),
            confidence: 0.0,
            cluster_id: -1,
            error: true,
        }
    }
}

/// Configuration for cluster router initialization
#[repr(C)]
pub struct ClusterRouterConfig {
    /// Number of clusters for K-means
    pub n_clusters: i32,
    /// Maximum iterations for K-means convergence
    pub max_iterations: i32,
    /// Cost-performance balance factor (0.0 = cost only, 1.0 = performance only)
    pub alpha: f32,
    /// Whether to use CPU (true) or GPU (false)
    pub use_cpu: bool,
}

impl Default for ClusterRouterConfig {
    fn default() -> Self {
        Self {
            n_clusters: 10,
            max_iterations: 100,
            alpha: 1.0, // Default: performance only
            use_cpu: false,
        }
    }
}

/// Experience record for FFI
#[repr(C)]
pub struct ExperienceRecordFFI {
    /// Pointer to embedding data
    pub embedding: *const f32,
    /// Length of embedding
    pub embedding_len: i32,
    /// Pointer to model names (array of C strings)
    pub model_names: *const *const c_char,
    /// Pointer to model scores (parallel array)
    pub model_scores: *const f32,
    /// Number of models
    pub num_models: i32,
}

/// K-means clustering implementation using candle-core
///
/// Reference: https://raw.githubusercontent.com/vishpat/candle-coursera-ml/refs/heads/main/k-means/src/main.rs
fn kmeans_fit(
    data: &Tensor,
    n_clusters: usize,
    max_iterations: usize,
    device: &Device,
) -> Result<(Tensor, Vec<usize>), candle_core::Error> {
    let (n_samples, _n_features) = data.dims2()?;

    if n_samples < n_clusters {
        return Err(candle_core::Error::Msg(format!(
            "Number of samples ({}) must be >= n_clusters ({})",
            n_samples, n_clusters
        )));
    }

    // Initialize cluster centers using random selection (K-means++ simplified)
    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let initial_indices: Vec<u32> = indices[..n_clusters].iter().map(|&x| x as u32).collect();
    let indices_tensor = Tensor::from_slice(&initial_indices, (n_clusters,), device)?;
    let mut centers = data.index_select(&indices_tensor, 0)?;

    let mut cluster_assignments = vec![0usize; n_samples];

    for _iter in 0..max_iterations {
        // Compute distances from each point to each center
        // Using broadcasting: data[n, d] - centers[k, d] -> dist[n, k]
        let data_expanded = data.unsqueeze(1)?; // [n, 1, d]
        let centers_expanded = centers.unsqueeze(0)?; // [1, k, d]
        let diff = data_expanded.broadcast_sub(&centers_expanded)?; // [n, k, d]
        let sq_diff = diff.sqr()?;
        let distances = sq_diff.sum(2)?; // [n, k] - squared distances

        // Assign each point to nearest cluster
        let assignments_tensor = distances.argmin(1)?; // [n]
        let new_assignments: Vec<usize> = assignments_tensor
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize)
            .collect();

        // Check for convergence
        let converged = new_assignments == cluster_assignments;
        cluster_assignments = new_assignments;

        if converged {
            break;
        }

        // Update cluster centers
        let mut new_centers = Vec::with_capacity(n_clusters);
        for k in 0..n_clusters {
            // Find indices of points assigned to cluster k
            let cluster_indices: Vec<u32> = cluster_assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| if c == k { Some(i as u32) } else { None })
                .collect();

            if cluster_indices.is_empty() {
                // Keep old center if cluster is empty
                let old_center = centers.narrow(0, k, 1)?;
                new_centers.push(old_center);
            } else {
                let indices_tensor =
                    Tensor::from_slice(&cluster_indices, (cluster_indices.len(),), device)?;
                let cluster_points = data.index_select(&indices_tensor, 0)?;
                let mean = cluster_points.mean(0)?;
                new_centers.push(mean.unsqueeze(0)?);
            }
        }

        centers = Tensor::cat(&new_centers, 0)?;
    }

    Ok((centers, cluster_assignments))
}

/// Compute cosine similarity between query and cluster centers
fn find_nearest_cluster(
    query_embedding: &Tensor,
    cluster_centers: &Tensor,
) -> Result<(usize, f32), candle_core::Error> {
    // Normalize query
    let query_norm = query_embedding.sqr()?.sum_all()?.sqrt()?;
    let query_normalized = query_embedding.broadcast_div(&query_norm)?;

    // Normalize centers
    let centers_norm = cluster_centers.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
    let centers_normalized = cluster_centers.broadcast_div(&centers_norm)?;

    // Compute cosine similarity: query @ centers.T
    let similarities = query_normalized.matmul(&centers_normalized.t()?)?;
    let similarities_1d = similarities.squeeze(0)?;

    // Find max similarity
    let max_idx = similarities_1d.argmax(0)?;
    let max_sim = similarities_1d.max(0)?;

    let cluster_id = max_idx.to_scalar::<i64>()? as usize;
    let similarity = max_sim.to_scalar::<f32>()?;

    Ok((cluster_id, similarity))
}

/// L2 normalize embeddings (critical for cosine similarity)
fn l2_normalize(embeddings: &Tensor) -> Result<Tensor, candle_core::Error> {
    // Compute L2 norm per row: sqrt(sum(x^2))
    let sq = embeddings.sqr()?;
    let row_sums = sq.sum(1)?; // [n_samples]
    let norms = row_sums.sqrt()?.unsqueeze(1)?; // [n_samples, 1]
    
    // Avoid division by zero - add small epsilon
    let eps = Tensor::full(1e-8f32, norms.shape(), norms.device())?;
    let safe_norms = norms.maximum(&eps)?;
    
    // Normalize: x / ||x||
    embeddings.broadcast_div(&safe_norms)
}

/// Train cluster router from experience data
///
/// This function:
/// 1. L2 normalizes the embeddings (required for cosine similarity)
/// 2. Runs K-means clustering on the normalized experience embeddings
/// 3. Computes average model performance per cluster
/// 4. Selects best model for each cluster based on alpha (performance vs cost)
fn train_cluster_router(
    experience_data: &[ExperienceRecord],
    model_costs: &HashMap<String, f32>,
    config: &ClusterRouterConfig,
    device: &Device,
) -> Result<ClusterRouterState, String> {
    if experience_data.is_empty() {
        return Err("Experience data is empty".to_string());
    }

    let embedding_dim = experience_data[0].embedding.len();
    let n_samples = experience_data.len();
    let n_clusters = (config.n_clusters as usize).min(n_samples);

    // Collect all model names
    let mut model_names_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for record in experience_data {
        for model_name in record.model_scores.keys() {
            model_names_set.insert(model_name.clone());
        }
    }
    let model_names: Vec<String> = model_names_set.into_iter().collect();

    // Build embedding matrix
    let flat_embeddings: Vec<f32> = experience_data
        .iter()
        .flat_map(|r| r.embedding.clone())
        .collect();

    let embeddings_tensor = Tensor::from_slice(&flat_embeddings, (n_samples, embedding_dim), device)
        .map_err(|e| format!("Failed to create embeddings tensor: {:?}", e))?;

    // L2 normalize embeddings (critical for cosine similarity / K-means)
    let embeddings_normalized = l2_normalize(&embeddings_tensor)
        .map_err(|e| format!("Failed to L2 normalize embeddings: {:?}", e))?;

    // Run K-means clustering on normalized embeddings
    let (cluster_centers, assignments) = kmeans_fit(
        &embeddings_normalized,
        n_clusters,
        config.max_iterations as usize,
        device,
    )
    .map_err(|e| format!("K-means failed: {:?}", e))?;

    // Compute model scores per cluster
    let mut cluster_model_scores: Vec<HashMap<String, Vec<f32>>> =
        vec![HashMap::new(); n_clusters];

    for (i, record) in experience_data.iter().enumerate() {
        let cluster_id = assignments[i];
        for (model_name, score) in &record.model_scores {
            cluster_model_scores[cluster_id]
                .entry(model_name.clone())
                .or_insert_with(Vec::new)
                .push(*score);
        }
    }

    // Compute average scores and select best model per cluster
    let mut cluster_avg_scores: Vec<HashMap<String, f32>> = Vec::with_capacity(n_clusters);
    let mut cluster_best_model: Vec<String> = Vec::with_capacity(n_clusters);

    for cluster_id in 0..n_clusters {
        let mut avg_scores: HashMap<String, f32> = HashMap::new();

        for (model_name, scores) in &cluster_model_scores[cluster_id] {
            if !scores.is_empty() {
                let avg = scores.iter().sum::<f32>() / scores.len() as f32;
                avg_scores.insert(model_name.clone(), avg);
            }
        }

        // Select best model considering cost
        let best_model = select_best_model_for_cluster(&avg_scores, model_costs, config.alpha);
        cluster_best_model.push(best_model);
        cluster_avg_scores.push(avg_scores);
    }

    Ok(ClusterRouterState {
        cluster_centers,
        cluster_best_model,
        cluster_model_scores: cluster_avg_scores,
        model_costs: model_costs.clone(),
        model_names,
        embedding_dim,
        n_clusters,
        device: device.clone(),
    })
}

/// Select best model for a cluster based on performance and cost
fn select_best_model_for_cluster(
    model_scores: &HashMap<String, f32>,
    model_costs: &HashMap<String, f32>,
    alpha: f32,
) -> String {
    if model_scores.is_empty() {
        return String::new();
    }

    // Normalize costs to 0-1 range
    let max_cost = model_costs
        .values()
        .cloned()
        .fold(1.0f32, |a, b| a.max(b));

    let mut best_model = String::new();
    let mut best_score = f32::NEG_INFINITY;

    for (model_name, &performance) in model_scores {
        let cost = model_costs.get(model_name).copied().unwrap_or(1.0);
        let normalized_cost = cost / max_cost;

        // Combined score: alpha * performance - (1 - alpha) * cost
        let combined_score = alpha * performance - (1.0 - alpha) * normalized_cost;

        if combined_score > best_score {
            best_score = combined_score;
            best_model = model_name.clone();
        }
    }

    best_model
}

// ============================================================================
// FFI Functions
// ============================================================================

/// Initialize cluster router with experience data
///
/// # Safety
/// - `experience_records` must be a valid pointer to an array of ExperienceRecordFFI
/// - `num_records` must be the correct count
/// - `model_names` and `model_costs` must be parallel arrays of valid C strings/floats
/// - `config` must be a valid pointer to ClusterRouterConfig
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn init_cluster_router(
    experience_records: *const ExperienceRecordFFI,
    num_records: i32,
    model_cost_names: *const *const c_char,
    model_cost_values: *const f32,
    num_model_costs: i32,
    config: *const ClusterRouterConfig,
) -> i32 {
    if experience_records.is_null() || config.is_null() {
        eprintln!("Error: null pointer passed to init_cluster_router");
        return -1;
    }

    if num_records <= 0 {
        eprintln!("Error: num_records must be positive");
        return -1;
    }

    // Parse config
    let cfg = unsafe { &*config };

    // Determine device
    let device = if cfg.use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    // Parse model costs
    let mut model_costs: HashMap<String, f32> = HashMap::new();
    if !model_cost_names.is_null() && !model_cost_values.is_null() && num_model_costs > 0 {
        unsafe {
            for i in 0..num_model_costs as isize {
                let name_ptr = *model_cost_names.offset(i);
                if !name_ptr.is_null() {
                    if let Ok(name) = CStr::from_ptr(name_ptr).to_str() {
                        let cost = *model_cost_values.offset(i);
                        model_costs.insert(name.to_string(), cost);
                    }
                }
            }
        }
    }

    // Parse experience records
    let mut experience_data: Vec<ExperienceRecord> = Vec::with_capacity(num_records as usize);

    unsafe {
        for i in 0..num_records as isize {
            let record = &*experience_records.offset(i);

            // Parse embedding
            if record.embedding.is_null() || record.embedding_len <= 0 {
                eprintln!("Error: invalid embedding in record {}", i);
                return -1;
            }
            let embedding: Vec<f32> =
                std::slice::from_raw_parts(record.embedding, record.embedding_len as usize)
                    .to_vec();

            // Parse model scores
            let mut model_scores: HashMap<String, f32> = HashMap::new();
            if !record.model_names.is_null()
                && !record.model_scores.is_null()
                && record.num_models > 0
            {
                for j in 0..record.num_models as isize {
                    let name_ptr = *record.model_names.offset(j);
                    if !name_ptr.is_null() {
                        if let Ok(name) = CStr::from_ptr(name_ptr).to_str() {
                            let score = *record.model_scores.offset(j);
                            model_scores.insert(name.to_string(), score);
                        }
                    }
                }
            }

            experience_data.push(ExperienceRecord {
                embedding,
                model_scores,
            });
        }
    }

    // Train the cluster router
    match train_cluster_router(&experience_data, &model_costs, cfg, &device) {
        Ok(state) => {
            if CLUSTER_ROUTER_STATE.set(state).is_err() {
                eprintln!("Warning: Cluster router already initialized");
            }
            println!(
                "INFO: Cluster router initialized with {} clusters, {} experience records",
                cfg.n_clusters, num_records
            );
            0
        }
        Err(e) => {
            eprintln!("Error: Failed to train cluster router: {}", e);
            -1
        }
    }
}

/// Route a query to the best model based on cluster membership
///
/// # Safety
/// - `query_embedding` must be a valid pointer to f32 array
/// - `embedding_len` must match the trained embedding dimension
/// - `result` must be a valid pointer to ClusterRouteResult
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn route_query(
    query_embedding: *const f32,
    embedding_len: i32,
    result: *mut ClusterRouteResult,
) -> i32 {
    if query_embedding.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to route_query");
        return -1;
    }

    let state = match CLUSTER_ROUTER_STATE.get() {
        Some(s) => s,
        None => {
            eprintln!("Error: Cluster router not initialized");
            unsafe {
                (*result) = ClusterRouteResult::default();
            }
            return -1;
        }
    };

    if embedding_len as usize != state.embedding_dim {
        eprintln!(
            "Error: embedding dimension mismatch ({} vs {})",
            embedding_len, state.embedding_dim
        );
        unsafe {
            (*result) = ClusterRouteResult::default();
        }
        return -1;
    }

    // Create query tensor
    let query_vec: Vec<f32> =
        unsafe { std::slice::from_raw_parts(query_embedding, embedding_len as usize).to_vec() };

    let query_tensor =
        match Tensor::from_slice(&query_vec, (1, state.embedding_dim), &state.device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error: Failed to create query tensor: {:?}", e);
                unsafe {
                    (*result) = ClusterRouteResult::default();
                }
                return -1;
            }
        };

    // Find nearest cluster
    let (cluster_id, similarity) = match find_nearest_cluster(&query_tensor, &state.cluster_centers)
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Failed to find nearest cluster: {:?}", e);
            unsafe {
                (*result) = ClusterRouteResult::default();
            }
            return -1;
        }
    };

    // Get best model for cluster
    let best_model = if cluster_id < state.cluster_best_model.len() {
        state.cluster_best_model[cluster_id].clone()
    } else {
        eprintln!("Error: Invalid cluster ID {}", cluster_id);
        unsafe {
            (*result) = ClusterRouteResult::default();
        }
        return -1;
    };

    // Convert model name to C string
    let model_cstring = match CString::new(best_model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: Failed to create model name string: {:?}", e);
            unsafe {
                (*result) = ClusterRouteResult::default();
            }
            return -1;
        }
    };

    unsafe {
        (*result) = ClusterRouteResult {
            model_name: model_cstring.into_raw(),
            confidence: similarity,
            cluster_id: cluster_id as i32,
            error: false,
        };
    }

    0
}

/// Get the number of clusters in the trained router
///
/// # Returns
/// Number of clusters, or -1 if not initialized
#[no_mangle]
pub extern "C" fn get_cluster_count() -> i32 {
    match CLUSTER_ROUTER_STATE.get() {
        Some(state) => state.n_clusters as i32,
        None => -1,
    }
}

/// Check if cluster router is initialized
///
/// # Returns
/// 1 if initialized, 0 if not
#[no_mangle]
pub extern "C" fn is_cluster_router_initialized() -> i32 {
    if CLUSTER_ROUTER_STATE.get().is_some() {
        1
    } else {
        0
    }
}

/// Get model scores for a specific cluster
///
/// # Safety
/// - `cluster_id` must be within valid range
/// - `model_names_out` and `scores_out` must be valid pointers with enough capacity
/// - `capacity` must indicate the array capacity
///
/// # Returns
/// Number of models written, or -1 on error
#[no_mangle]
pub extern "C" fn get_cluster_model_scores(
    cluster_id: i32,
    model_names_out: *mut *mut c_char,
    scores_out: *mut f32,
    capacity: i32,
) -> i32 {
    if model_names_out.is_null() || scores_out.is_null() {
        return -1;
    }

    let state = match CLUSTER_ROUTER_STATE.get() {
        Some(s) => s,
        None => return -1,
    };

    if cluster_id < 0 || cluster_id as usize >= state.n_clusters {
        return -1;
    }

    let scores = &state.cluster_model_scores[cluster_id as usize];
    let count = scores.len().min(capacity as usize);

    unsafe {
        for (i, (model_name, score)) in scores.iter().take(count).enumerate() {
            if let Ok(cstr) = CString::new(model_name.as_str()) {
                *model_names_out.add(i) = cstr.into_raw();
                *scores_out.add(i) = *score;
            }
        }
    }

    count as i32
}

/// Free a cluster route result
///
/// # Safety
/// - `result` must be a valid pointer to ClusterRouteResult
#[no_mangle]
pub extern "C" fn free_cluster_route_result(result: *mut ClusterRouteResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let res = &mut *result;
        if !res.model_name.is_null() {
            let _ = CString::from_raw(res.model_name);
            res.model_name = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let device = Device::Cpu;

        // Create simple 2D data with 2 obvious clusters
        let data = vec![
            0.0f32, 0.0, 0.1, 0.1, 0.2, 0.0, // Cluster 1 near origin
            10.0, 10.0, 10.1, 10.1, 10.2, 10.0, // Cluster 2 far away
        ];
        let tensor = Tensor::from_slice(&data, (6, 2), &device).unwrap();

        let (centers, assignments) = kmeans_fit(&tensor, 2, 100, &device).unwrap();

        // Verify we have 2 clusters
        assert_eq!(centers.dims()[0], 2);

        // Verify first 3 points are in same cluster, last 3 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn test_select_best_model() {
        let mut model_scores = HashMap::new();
        model_scores.insert("model_a".to_string(), 0.9);
        model_scores.insert("model_b".to_string(), 0.8);
        model_scores.insert("model_c".to_string(), 0.7);

        let mut model_costs = HashMap::new();
        model_costs.insert("model_a".to_string(), 10.0);
        model_costs.insert("model_b".to_string(), 5.0);
        model_costs.insert("model_c".to_string(), 1.0);

        // Performance only (alpha = 1.0)
        let best = select_best_model_for_cluster(&model_scores, &model_costs, 1.0);
        assert_eq!(best, "model_a"); // Highest performance

        // Cost only (alpha = 0.0)
        let best = select_best_model_for_cluster(&model_scores, &model_costs, 0.0);
        assert_eq!(best, "model_c"); // Lowest cost

        // Balanced (alpha = 0.5)
        // Scores: model_a = 0.5*0.9 - 0.5*1.0 = -0.05
        //         model_b = 0.5*0.8 - 0.5*0.5 = 0.15
        //         model_c = 0.5*0.7 - 0.5*0.1 = 0.30
        // model_c wins at alpha=0.5 because cost savings outweigh small performance loss
        let best = select_best_model_for_cluster(&model_scores, &model_costs, 0.5);
        assert_eq!(best, "model_c");
        
        // High performance bias (alpha = 0.8)
        // Scores: model_a = 0.8*0.9 - 0.2*1.0 = 0.52
        //         model_b = 0.8*0.8 - 0.2*0.5 = 0.54
        //         model_c = 0.8*0.7 - 0.2*0.1 = 0.54
        // model_b and model_c tie, but order depends on hash iteration
        let best = select_best_model_for_cluster(&model_scores, &model_costs, 0.8);
        assert!(best == "model_b" || best == "model_c"); // Either is valid
    }
}

