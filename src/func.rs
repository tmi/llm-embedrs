// utility functions for operating tensors

// TODO ndarray or tensors? Make the types neater!
use ndarray::{ArrayView, ArrayBase, OwnedRepr, Axis, IxDynImpl, Dim, s};

pub fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let mag_a = f32::sqrt(a.iter().map(|e| e*e).sum());
    let mag_b = f32::sqrt(b.iter().map(|e| e*e).sum());
    dot / (mag_a * mag_b)
}


pub fn max_pooling(outputs: ArrayView<f32, Dim<IxDynImpl>>, attention_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>) -> Result<Vec<f32>, Box<dyn std::error::Error>>  {
    let outputs_shape = outputs.shape();
    let outputs_shape_broadcastable = (outputs_shape[2], outputs_shape[0], outputs_shape[1]);
    let outputs_view = outputs.to_shape((outputs_shape[0], outputs_shape[1], outputs_shape[2]))?;
    let attention_mx = attention_mask.broadcast(outputs_shape_broadcastable).unwrap().permuted_axes([1, 2, 0]).mapv(|x| x as f32);
    let outputs_masked = &outputs_view * &attention_mx;
    let outputs_sum = outputs_masked.sum_axis(Axis(1));
    let attention_mx_sum = attention_mx.sum_axis(Axis(1));
    let sentence_embeddings = outputs_sum / attention_mx_sum; // consider clamp min=1e-9 to not have 0s down
    let l2n = sentence_embeddings.slice(s![0, ..]).map(|x| x.powi(2)).sum().sqrt();
    let sentence_embeddings_n = (&sentence_embeddings.slice(s![0, ..])) / l2n;

    Ok(sentence_embeddings_n.to_vec())
}
