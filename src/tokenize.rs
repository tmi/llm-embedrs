// loading and invoking tokenizer

use tokenizers::Tokenizer;
use ndarray::{Array, ArrayBase, OwnedRepr, Dim};

fn load_tokenizer(tokenizer_path: &str) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).or_else(
        |err| Err(format!("Failed to load tokenizer from {} with {}", tokenizer_path, err)) 
    )?;
    Ok(tokenizer)
}

type TokenArray = ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>;

#[derive(Debug)]
pub struct Tokens {
    pub ids: TokenArray,
    pub attention_mask: TokenArray,
    pub type_ids: TokenArray,
}

pub fn tokenize(tokenizer_path: &str, text: &str) -> Result<Tokens, Box<dyn std::error::Error>> {
    let tokenizer = load_tokenizer(tokenizer_path)?;

    // TODO tokenizer is set to output max length 128. We handle padding with attention mask, but what about longer inputs?
    let encoding = tokenizer.encode(text, true).expect("Failed to encode");

    let ids = Array::from_iter(encoding.get_ids().iter().map(|&x| x as i64));
    let shape = (1, ids.shape()[0]);
    let ids = ids.to_shape(shape)?.to_owned();
    let attention_mask = Array::from_iter(encoding.get_attention_mask().iter().map(|&x| x as i64)).to_shape(shape)?.to_owned();
    let type_ids = Array::from_iter(encoding.get_type_ids().iter().map(|&x| x as i64)).to_shape(shape)?.to_owned();

    Ok(Tokens {ids, attention_mask, type_ids } )
}
