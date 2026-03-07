//! Shared pooling strategies for embedding extraction.

/// Mean pooling: average all token embeddings
pub fn pool_mean(hidden: &[f32], num_tokens: usize, hidden_size: usize) -> Vec<f32> {
    if num_tokens == 0 || hidden_size == 0 {
        return Vec::new();
    }
    let mut result = vec![0.0f32; hidden_size];
    for t in 0..num_tokens {
        let offset = t * hidden_size;
        for (j, val) in result.iter_mut().enumerate() {
            *val += hidden[offset + j];
        }
    }
    let scale = 1.0 / num_tokens as f32;
    for val in &mut result {
        *val *= scale;
    }
    result
}

/// CLS pooling: use the first token's embedding
pub fn pool_cls(hidden: &[f32], hidden_size: usize) -> Vec<f32> {
    if hidden.len() < hidden_size {
        return Vec::new();
    }
    hidden[..hidden_size].to_vec()
}

/// Last token pooling: use the last token's embedding
pub fn pool_last(hidden: &[f32], num_tokens: usize, hidden_size: usize) -> Vec<f32> {
    if num_tokens == 0 || hidden_size == 0 {
        return Vec::new();
    }
    let offset = (num_tokens - 1) * hidden_size;
    hidden[offset..offset + hidden_size].to_vec()
}

/// L2-normalize a vector in-place
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_mean() {
        // 2 tokens, hidden_size=3: [[1,2,3],[4,5,6]] → mean [2.5, 3.5, 4.5]
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = pool_mean(&hidden, 2, 3);
        assert_eq!(result, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_pool_mean_empty() {
        assert!(pool_mean(&[], 0, 3).is_empty());
        assert!(pool_mean(&[], 2, 0).is_empty());
    }

    #[test]
    fn test_pool_cls() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(pool_cls(&hidden, 3), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pool_cls_empty() {
        assert!(pool_cls(&[1.0], 3).is_empty());
    }

    #[test]
    fn test_pool_last() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(pool_last(&hidden, 2, 3), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
