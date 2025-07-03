mod db;
mod func;
mod onnx;
mod tokenize;

use crate::db::retrieve;
use crate::tokenize::tokenize;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    query: String,
    #[arg(short, long)]
    tokenizer: String,
    #[arg(short, long)]
    model: String,
    #[arg(short, default_value_t = 1)]
    n: usize,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // * INPUT & CONFIG *
    let args = Args::parse();

    // * TOKENIZE *
    let tokens = tokenize(&args.tokenizer, &args.query)?;
    // * EMBEDD *
    let embedding = onnx::invoke(&args.model, tokens)?;
    // * RETRIEVE *
    let mut database = retrieve()?;
    // * SCORE *
    for entry in database.iter_mut() {
        let score = func::cosine_similarity(&embedding, &entry.embedding);
        entry.score = score;
    }
    // * EVALUATE *
    database.sort_by(|k2, k1| k1.score.partial_cmp(&k2.score).unwrap()); // NOTE k2, k1 causes reverse sort

    println!("{}", serde_json::to_string(&database[..args.n])?);

    Ok(())
}
