// onnx session & model invocation

use crate::tokenize::Tokens;
use crate::func::max_pooling;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

pub fn invoke(model_path: &str, tokens: Tokens) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // TODO check parameter sanity
    // TODO consider other runtimes
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    // TODO is this clone needed? Cant we feed views into the ort inputs?
    let attention_mask = tokens.attention_mask.clone();
    let outputs = model.run(ort::inputs![Tensor::from_array(tokens.ids)?, Tensor::from_array(tokens.attention_mask)?, Tensor::from_array(tokens.type_ids)?])?;
    let outputs_array = outputs["last_hidden_state"].try_extract_array()?;

    let transformed = max_pooling(outputs_array, attention_mask)?;

    Ok(transformed)
}
