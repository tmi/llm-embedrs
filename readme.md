# llm-embedrs

The [llm](https://github.com/simonw/llm) tool provides, among others, two useful commands:
* `llm embed <document>` to compute an embedding of the document, and store it to an sqlite database,
* `llm similar <query>` to compute an embedding of the query, and find the most similar documents in the database.

However, as the tool is in python, there is quite some overhead in execution -- e.g., on a collection of 70 documents, `llm similar` takes about 8 seconds on my puny laptop.

Thus I wrote this `llm-embedrs` tool in Rust, with ~40x speedup -- the same usecase being done in 0.2 seconds.
Nothing magical explains the speedup -- heavy lifting is done by onnx, it's just app startup & db read & bunch of dot products thats faster in Rust.

## Installation
```
cargo install --path .
```

## Usage
See `llm-embedrs --help`.
The output is in the same format as that of `llm similar`.

Currently, only the `llm similar` functionality is supported.
For `llm embed`, you best run the original tool, as the database schemata are compatible -- I didn't implement that yet because I use `similar` more often and in a blocking manner.

You will need to download a model such as [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  -- both the onnx file and the json of the tokenizer, and use the paths to these two as `-m` and `-t` parameters. 
It should work with other models -- as long as they accept the same input (tokens, attentio mask, token types) and produce the same output (last hidden state + assume to be max pooled afterwards), but this is the only one I tested.

## Missing
* functionality
  * implement `embed` 
  * implement collection filtering and other `similar` params
* convenience
  * make this a plugin for `llm` itself, both as a standalone executable and PyO3-exposed library
  * add integration to HF, so that models don't need explicit download
* performance
  * replace the manual `max pooling` implementation with some existing 3rd party
  * revisit the ndarray/arrow/tensor choice across `db`, `func`
  * run the scoring in parallel
  * proper setting of the onnx session, consider gpu runtime, candle backend, etc
* reliability
  * test that other models/tokenizers work as well
  * add tests

Open an issue if you actually find this project useful and would like to see something of the above implemented.
