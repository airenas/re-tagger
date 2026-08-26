use anyhow::{Context, Result};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use std::{fs, time::Instant};
use tokenizers::Tokenizer;

fn load_tags(path: &str) -> Result<Vec<String>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .map(str::to_owned)
        .collect())
}

fn predict(
    text: &str,
    tokenizer: &Tokenizer,
    session: &mut Session,
    tags: &[String],
    log: bool,
) -> Result<Vec<String>> {
    // Python:
    //
    // text.split("-")
    //
    // and then:
    //
    // tokenizer(words, is_split_into_words=True, ...)

    // let words: Vec<&str> = text.split('-').collect();

    // println!("Words: {:?}", words);

    // Important: encode the words as pre-tokenized input.
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();

    let ids = encoding.get_ids();
    if log {
        let tokens = encoding.get_tokens();
        let offsets = encoding.get_offsets();

        println!("Input IDs: {ids:?}");
        println!("Tokens: {tokens:?}");

    // ------------------------------------------------------------
    // Show token -> character mapping
    // ------------------------------------------------------------

        for ((token, offset), id) in tokens.iter().zip(offsets.iter()).zip(ids.iter()) {
            let (start, end) = *offset;

            let original = if start < end { &text[start..end] } else { "" };

            println!("{token:10} id={id:<6} chars=({start:2}, {end:2}) text={original:?}");
        }
    }

    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as i64)
        .collect();

    if log {
        println!("Input IDs: {:?}", input_ids);
    }

    let tokens = encoding.get_tokens();
    if log {
        println!("Tokens: {:?}", tokens);
    }

    let seq_len = input_ids.len();

    // ONNX inputs are:
    //
    // input_ids:      [batch, sequence]
    // attention_mask: [batch, sequence]
    let input_ids = Array2::from_shape_vec((1, seq_len), input_ids)?;

    let attention_mask = Array2::from_shape_vec((1, seq_len), attention_mask)?;

    let outputs = session.run(ort::inputs! {
        "input_ids" => Tensor::from_array(input_ids)?,
        "attention_mask" => Tensor::from_array(attention_mask)?,
    })?;

    // logits shape:
    //
    // [1, sequence, 537]
    //
    // Find the output named "logits".
    let logits = outputs
        .get("logits")
        .context("missing logits output")?
        .try_extract_tensor::<f32>()?;

    let (shape, data) = logits;

    if log {
        println!("Logits shape: {:?}", shape);
    }

    let num_labels = shape[2] as usize;

    let mut predicted = Vec::new();

    // encoding.word_ids() equivalent:
    //
    // We need only the FIRST subtoken of each word.
    //
    // `tokenizers` gives us word IDs.
    let word_ids = encoding.get_word_ids();

    let mut previous_word_id = None;

    for token_pos in 0..seq_len {
        let word_id = word_ids[token_pos];

        let Some(word_id) = word_id else {
            previous_word_id = None;
            continue;
        };

        // Skip subsequent subtokens of the same word.
        if Some(word_id) == previous_word_id {
            continue;
        }

        previous_word_id = Some(word_id);

        // logits[0, token_pos, :]
        let start = token_pos * num_labels;
        let end = start + num_labels;
        let row = &data[start..end];

        let predicted_id = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .context("empty logits")?;

        let tag = tags
            .get(predicted_id)
            .with_context(|| format!("invalid tag id {}", predicted_id))?;

        predicted.push(tag.clone());
    }

    Ok(predicted)
}

fn main() -> Result<()> {
    let model_dir = "../models";

    let model_path = format!("{}/model.int8.onnx", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tags_path = format!("{}/tags.txt", model_dir);

    println!("Loading tokenizer: {}", tokenizer_path);

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

    println!("Loading tags: {}", tags_path);

    let tags = load_tags(&tags_path)?;

    println!("Loaded {} tags", tags.len());

    println!("Loading ONNX model: {}", model_path);

    let mut session = Session::builder()?.commit_from_file(&model_path)?;

    println!("ONNX model loaded!");

    for input in session.inputs() {
        println!("input: {}", input.name());
    }

    for output in session.outputs() {
        println!("output: {}", output.name());
    }

    let text = "Mama Mama Mama mama su kasa kasa bulves.";

    println!("Predicting: {}", text);

    let predicted = predict(text, &tokenizer, &mut session, &tags, true)?;
    println!("Predicted: {:?}", predicted);

    let start = Instant::now();

    let c = 1000;
    for _ in 0..c {
        predict(text, &tokenizer, &mut session, &tags, false)?;
    }

    println!("{:?} sentences: {:?}", c, start.elapsed());
    println!("average: {:?}", start.elapsed() / c);

    Ok(())
}
