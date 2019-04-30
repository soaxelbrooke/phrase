#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;
#[macro_use] extern crate rocket_contrib;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
#[macro_use] extern crate log;
#[macro_use] extern crate lazy_static;

extern crate rusqlite;
extern crate env_logger;
// extern crate native_stemmers; // see https://crates.io/crates/rust-stemmers

use std::io::{BufReader, BufRead};
use regex::Regex;
use std::collections::{HashMap, VecDeque, HashSet};
use std::io::prelude::*;
use std::io::LineWriter;
use std::path::Path;
use std::fs::File;

use rocket_contrib::json::{Json, JsonValue};

const FILE_DELIM: char = '\t';

// URL regex
// [-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)

lazy_static! {
    static ref HEAD_UNIGRAM_IGNORES: HashSet<String> = vec!(
        "the", "a", "is", "and", "of", "to",
    ).iter().map(|s| s.to_string()).collect();

    static ref TAIL_UNIGRAM_IGNORES: HashSet<String> = vec!(
        "the", "a", "i", "is", "you", "and", "my", "so", "for",
    ).iter().map(|s| s.to_string()).collect();
}

fn parse_env<T: std::str::FromStr + Clone + std::fmt::Debug>(key: &str, default: T) -> T {
    std::env::var(key).map(|s| {
        match T::from_str(&s) {
            Ok(v) => v,
            Err(_err) => {
                error!("Couldn't parse env var {}", key);
                default.to_owned()
            },
        }
    }).unwrap_or(default.to_owned())
}

lazy_static! {
    static ref PRUNE_AT: usize = parse_env("PRUNE_AT", 5_000_000);
    static ref PRUNE_TO: usize = parse_env("PRUNE_TO", 1_000_000);
    static ref MAX_NGRAM: usize = parse_env("MAX_NGRAM", 4);
    static ref MIN_COUNT: i64 = parse_env("MIN_COUNT", 5);
    static ref MIN_SCORE: f64 = parse_env("MIN_SCORE", 0.1f64);
    static ref MAX_EXPORT: u32 = parse_env("MAX_EXPORT", 250_000);

    static ref TOKEN_REGEX: Regex = Regex::new(&std::env::var("TOKEN_REGEX").unwrap_or(r"[\w+'’]+".to_string())).unwrap();
    static ref CHUNK_SPLIT_REGEX: Regex = Regex::new(&std::env::var("CHUNK_SPLIT_REGEX").unwrap_or(r"[\.\?!\(\);]+".to_string())).unwrap();
}

lazy_static! {
    static ref SCORES: HashMap<Option<String>, HashMap<Vec<String>, f64>> = {
        let mut label_ngrams = HashMap::new();
        read_partition_scores_for_labels(&Some(list_score_labels().unwrap()), &mut label_ngrams).expect("Unable to read partitions for labels");
        label_ngrams
    };
}

#[derive(Deserialize, Serialize, Debug)]
struct Document {
    label: Option<String>,
    text: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AnalyzedDocument {
    label: Option<String>,
    text: String,
    ngrams: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ApiAnalyzeRequest {
    documents: Vec<Document>
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String
}

#[derive(Serialize)]
struct ApiListLabelsResponse {
    labels: Vec<Option<String>>
}

fn analyze_text(text: &String, scores: &HashMap<Vec<String>, f64>, max_ngram: &usize) -> Vec<String> {
    let mut significant_ngrams: Vec<String> = vec!();

    for chunk in CHUNK_SPLIT_REGEX.split(&text) {
        let mut token_queues: Vec<VecDeque<String>> = Vec::new();
        for i in 1..max_ngram+1 {
            token_queues.push(VecDeque::with_capacity(i));
        }
        for token in TOKEN_REGEX.find_iter(chunk) {
            let token_string = token.as_str().to_string();
            for i in 1..max_ngram+1 {
                if let Some(queue) = token_queues.get_mut(i - 1) {
                    queue.push_back(token_string.to_owned());
                    if queue.len() > i {
                        queue.pop_front();
                    }
                    if queue.len() == i {
                        let queue: Vec<String> = queue.iter().map(|s| s.to_string()).collect();
                        if let Some(score) = scores.get(&queue) {
                            if score > &MIN_SCORE {
                                let ngram: Vec<String> = queue.iter().map(|s| s.to_string()).collect();
                                significant_ngrams.push(ngram.join(" "));
                            }
                        }
                    }
                }
            }
        }
    }

    significant_ngrams.sort();
    significant_ngrams.dedup();

    significant_ngrams
}

#[post("/analyze", data = "<data>")]
fn api_analyze(data: Json<ApiAnalyzeRequest>) -> JsonValue {
    let max_ngram = MAX_NGRAM.clone();
    let analyzed_docs: Vec<AnalyzedDocument> = data.0.documents.iter().map(|d| {
        let significant_terms: Vec<String> = if let Some(scores) = SCORES.get(&d.label) {
            analyze_text(&d.text, scores, &max_ngram)
        } else {
            vec!()
        };
        AnalyzedDocument {
            label: d.label.to_owned(),
            text: d.text.to_owned(),
            ngrams: significant_terms,
        }
    }).collect();
    json!(analyzed_docs)
}

#[get("/labels")]
fn api_list_labels() -> JsonValue {
    if let Ok(labels) = list_score_labels() {
        json!(ApiListLabelsResponse { labels: labels })
    } else {
        json!(ErrorResponse { error: "Couldn't list labels.".to_string() })
    }
}

fn serve(host: &str, port: u16) {
    println!("Starting phrase server");
    let config = rocket::Config::build(rocket::config::Environment::Development)
        .port(port).address(host).finalize().expect("Couldn't create config.");
    rocket::custom(config)
        .mount("/", routes![api_list_labels, api_analyze])
        .launch();
}

fn read_partition_counts_for_labels(labels: &Option<Vec<Option<String>>>, label_ngrams: &mut HashMap<Option<String>, HashMap<Vec<String>, i64>>) -> std::io::Result<()> {
    if let Some(labels) = labels {
        for label in labels {
            let ngrams = match read_partition_counts(&label.as_ref()) {
                Ok(Some(ngrams)) => ngrams,
                Ok(None) => HashMap::new(),
                Err(err) => return Err(err),
            };
            label_ngrams.insert(label.to_owned(), ngrams);
        }
    } else {
        let ngrams = match read_partition_counts(&None) {
            Ok(Some(ngrams)) => ngrams,
            Ok(None) => HashMap::new(),
            Err(err) => return Err(err),
        };
        label_ngrams.insert(None, ngrams);
    }
    Ok(())
}

fn read_partition_scores_for_labels(labels: &Option<Vec<Option<String>>>, label_scores: &mut HashMap<Option<String>, HashMap<Vec<String>, f64>>) -> std::io::Result<()> {
    if let Some(labels) = labels {
        for label in labels {
            let scores = match read_partition_scores(&label.as_ref()) {
                Ok(Some(scores)) => scores,
                Ok(None) => HashMap::new(),
                Err(err) => return Err(err),
            };
            label_scores.insert(label.to_owned(), scores);
        }
    } else {
        let scores = match read_partition_scores(&None) {
            Ok(Some(scores)) => scores,
            Ok(None) => HashMap::new(),
            Err(err) => return Err(err),
        };
        label_scores.insert(None, scores);
    }
    Ok(())
}

fn update_phrase_models(label: Option<String>, documents: &mut Vec<String>) -> std::io::Result<()> {
    documents.sort();
    documents.dedup();
    let mut label_ngrams: HashMap<Option<String>, HashMap<Vec<String>, i64>> = HashMap::new();
    read_partition_counts_for_labels(&label.clone().map(|l| vec!(Some(l))), &mut label_ngrams)?;
    let mut ngrams = label_ngrams.get_mut(&label).unwrap();
    merge_ngrams_into(&count_ngrams(&documents), &mut ngrams);
    write_partition_counts(label.as_ref(), &ngrams)?;
    Ok(())
}

fn update_phrase_models_from_labeled_documents(labeled_documents: &mut Vec<LabeledDocument>) -> std::io::Result<()> {
    let mut groups: HashMap<Option<String>, Vec<String>> = HashMap::new();
    for labeled_document in labeled_documents {
        if groups.contains_key(&labeled_document.label) {
            groups.get_mut(&labeled_document.label).unwrap().push(labeled_document.text.clone());
        } else {
            assert_label_valid(&labeled_document.label.to_owned().as_ref());
            groups.insert(labeled_document.label.clone(), vec!(labeled_document.text.clone()));
        }
    }

    debug!("Counting ngrams for labels: {:?}", groups.keys());

    for (label, documents) in groups.iter_mut() {
        debug!("Counting ngrams for label: {:?}", label);
        update_phrase_models(label.clone(), documents)?;
    }

    Ok(())
}

fn merge_ngrams_into(from: &HashMap<Vec<String>, i64>, into: &mut HashMap<Vec<String>, i64>) {
    for (ngram, count) in from {
        if let Some(into_count) = into.get_mut(ngram) {
            *into_count += count;
        } else {
            into.insert(ngram.clone(), count.clone());
        }
    }
}

fn count_ngrams_into(documents: &Vec<String>, ngrams: &mut HashMap<Vec<String>, i64>) {
    let max_ngram = MAX_NGRAM.clone();
    let mut doc_count = 0;
    for document in documents {
        count_document_ngrams(&document, ngrams, &max_ngram);
        doc_count += 1;
        if (doc_count % 1000) == 0 {
            prune_ngrams(ngrams);
        }
    }
}

fn count_ngrams(documents: &Vec<String>) -> HashMap<Vec<String>, i64> {
    let mut ngrams = HashMap::new();
    debug!("Counting ngrams for {} documents.", documents.len());
    count_ngrams_into(documents, &mut ngrams);
    ngrams
}

fn count_document_ngrams(document: &String, ngrams: &mut HashMap<Vec<String>, i64>, max_ngram: &usize) {
    let mut unique_ngrams: HashSet<Vec<String>> = HashSet::new();

    for chunk in CHUNK_SPLIT_REGEX.split(&document) {
        let mut token_queues: Vec<VecDeque<String>> = Vec::new();
        for i in 1..max_ngram+1 {
            token_queues.push(VecDeque::with_capacity(i));
        }
        for token in TOKEN_REGEX.find_iter(chunk) {
            let token_string = token.as_str().to_string();
            for i in 1..max_ngram+1 {
                if let Some(queue) = token_queues.get_mut(i - 1) {
                    queue.push_back(token_string.to_owned());
                    if queue.len() > i {
                        queue.pop_front();
                    }
                    if queue.len() == i {
                        let queue: Vec<String> = queue.iter().map(|s| s.to_string()).collect();
                        unique_ngrams.insert(queue);
                    }
                }
            }
        }
    }

    for ngram in unique_ngrams.iter() {
        let ngram = ngram.to_owned();
        if let Some(count) = ngrams.get_mut(&ngram) {
            *count += 1;
        } else {
            ngrams.insert(ngram, 1);
        }
    }
}

fn prune_ngrams(ngrams: &mut HashMap<Vec<String>, i64>) {
    let ngrams_len = ngrams.len();
    if ngrams_len > PRUNE_AT.clone() {
        debug!("Pruning ngrams of length {}.", ngrams_len);
        let mut ngram_pairs: Vec<(&Vec<String>, &i64)> = ngrams.iter().collect();
        ngram_pairs.sort_by(|a, b| b.1.cmp(a.1));
        let ngrams_to_prune: Vec<Vec<String>> = ngram_pairs.iter().map(|(ngram, _count)| ngram.to_owned().to_owned()).collect();
        for ngram in ngrams_to_prune {
            ngrams.remove(&ngram);
        }
        debug!("Done pruning.");
    }
}

fn read_partition_counts(label: &Option<&String>) -> std::io::Result<Option<HashMap<Vec<String>, i64>>> {
    debug!("Reading counts for label {:?}", label);
    let path_str: String = match label {
        Some(label) => format!("data/counts_label={}.txt", label),
        None => String::from("data/counts_root.txt"),
    };
    let path = Path::new(&path_str);
    if path.exists() {
        let file = File::open(path)?;
        let file = BufReader::new(file);
        let mut ngrams = HashMap::new();
        for line in file.lines() {
            if let Ok(line) = line {
                let mut split = line.split(FILE_DELIM);
                let ngram: Option<Vec<String>> = split.next().map(|s| s.split(' ').map(|s| s.to_string()).collect());
                let count = split.next().map(|s| s.parse::<i64>());
                if let Some(ngram) = ngram {
                    if let Some(Ok(count)) = count {
                        ngrams.insert(ngram, count);
                    }
                }
            }
        }

        Ok(Some(ngrams))
    } else {
        Ok(None)
    }
}

fn read_partition_scores(label: &Option<&String>) -> std::io::Result<Option<HashMap<Vec<String>, f64>>> {
    let path_str: String = match label {
        Some(label) => format!("data/scores_label={}.txt", label),
        None => String::from("data/scores_root.txt"),
    };
    let path = Path::new(&path_str);
    if path.exists() {
        let file = File::open(path)?;
        let file = BufReader::new(file);
        let mut scores = HashMap::new();
        for line in file.lines() {
            if let Ok(line) = line {
                let mut split = line.split(FILE_DELIM);
                let ngram: Option<Vec<String>> = split.next().map(|s| s.split(' ').map(|s| s.to_string()).collect());
                let count = split.next().map(|s| s.parse::<f64>());
                if let Some(ngram) = ngram {
                    if let Some(Ok(count)) = count {
                        scores.insert(ngram, count);
                    }
                }
            }
        }

        Ok(Some(scores))
    } else {
        Ok(None)
    }
}

fn write_partition_counts(label: Option<&String>, ngrams: &HashMap<Vec<String>, i64>) -> std::io::Result<()> {
    debug!("Writing partition {:?}.", &label);
    let mut counts: Vec<(&Vec<String>, &i64)> = ngrams.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    let file = match label {
        Some(label) => File::create(format!("data/counts_label={}.txt", label))?,
        None => File::create("data/counts_root.txt")?,
    };
    let mut file = LineWriter::new(file);
    for (ngram, count) in counts.iter().take(PRUNE_TO.clone()) {
        let phrase = ngram.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(" "); 
        writeln!(file, "{}{}{}", phrase, FILE_DELIM, count)?;
    }
    file.flush()?;
    Ok(())
}

fn count_stdin(label: Option<String>, is_csv: bool) {
    if is_csv {
        let mut reader = csv::Reader::from_reader(std::io::stdin());
        let mut labeled_documents: Vec<LabeledDocument> = vec!();

        for result in reader.deserialize() {
            let labeled_document: LabeledDocument = result.expect("failed to parse line");
            labeled_documents.push(labeled_document);
        }

        update_phrase_models_from_labeled_documents(&mut labeled_documents).expect("Failed to update phrase models.");
    } else {
        let stdin = std::io::stdin();
        let mut documents: Vec<String> = stdin.lock().lines().map(|s| s.expect("Coulnd't read line")).collect();
        update_phrase_models(label, &mut documents).expect("Failed to update phrase models");
    }
}

#[derive(Deserialize)]
struct LabeledDocument {
    label: Option<String>,
    text: String,
}

fn count_file(path: &str, label: Option<String>, is_csv: bool) {
    if is_csv {
        let mut reader = csv::Reader::from_path(path).expect("Couldn't open CSV");
        let mut labeled_documents: Vec<LabeledDocument> = vec!();

        for result in reader.deserialize() {
            let labeled_document: LabeledDocument = result.expect("failed to parse line");
            labeled_documents.push(labeled_document);
        }

        update_phrase_models_from_labeled_documents(&mut labeled_documents).expect("Failed to update phrase models.");
    } else {
        let file = std::fs::File::open(path).unwrap();
        let mut documents: Vec<String> = BufReader::new(file).lines().map(|s| s.expect("Couldn't read line")).collect();
        update_phrase_models(label, &mut documents).expect("Failed to update phrase models");
    };
}

fn cmd_count(path: &str, label: Option<String>, is_csv: bool) {
    if is_csv && label.is_some() {
        error!("Cannot specify label and provide a CSV");
        std::process::exit(1);
    }
    std::fs::create_dir_all("data").expect("Failed to ensure data directory existence.");
    if path == "-" {
        count_stdin(label, is_csv);
    } else {
        count_file(path, label, is_csv);
    };
}

fn cmd_transform(input: String, is_csv: bool, label: Option<String>, delim: String, output: Option<String>) {
    if is_csv && label.is_some() {
        error!("Cannot specify label and provide a CSV");
        std::process::exit(1);
    }

    if is_csv {
        transform_csv(input, delim, output);
    } else {
        transform_standard(input, label, delim, output);
    }
}

fn transform_csv(input: String, delim: String, output: Option<String>) {
    if input == "-" {
        transform_csv_inner(&mut csv::Reader::from_reader(std::io::stdin()), delim, output);
    } else {
        transform_csv_inner(&mut csv::Reader::from_path(input).expect("Couldn't read provided input file"), delim, output);
    }
}

fn transform_csv_inner<T: std::io::Read>(csv_reader: &mut csv::Reader<T>, delim: String, output: Option<String>) {
    if let Some(path) = output {
        transform_csv_inner_2(csv_reader, &mut csv::Writer::from_path(&path).expect("Couldn't open output file for writing"), &delim);
    } else {
        transform_csv_inner_2(csv_reader, &mut csv::Writer::from_writer(std::io::stdout()), &delim);
    };
    
}

fn transform_csv_inner_2<A: std::io::Read, B: std::io::Write>(csv_reader: &mut csv::Reader<A>, csv_writer: &mut csv::Writer<B>, delim: &String) {
    for result in csv_reader.deserialize() {
        let document: Document = result.expect("Unable to parse csv document");
        let transformed = transform_text(delim, &document.label, &document.text);
        csv_writer.serialize(Document { label: document.label, text: transformed}).expect("Couldn't write CSV output");
    }
    csv_writer.flush().expect("Couldn't flush CSV output buffer");
}

fn transform_standard(input: String, label: Option<String>, delim: String, output: Option<String>) {
    if let Some(path) = output {
        if &input == "-" {
            transform_standard_inner(label, delim, &mut File::create(path).expect("Couldn't read input path"), BufReader::new(std::io::stdin()));
        } else {
            transform_standard_inner(label, delim, &mut File::create(path).expect("Couldn't read input path"), BufReader::new(File::open(input).expect("Couldn't open input file for reading")));
        }
    } else {
        if &input == "-" {
            transform_standard_inner(label, delim, &mut std::io::stdout(), BufReader::new(std::io::stdin()));
        } else {
            transform_standard_inner(label, delim, &mut std::io::stdout(), BufReader::new(File::open(input).expect("Couldn't open input file for reading")));
        }
    }
}

fn transform_standard_inner<T: std::io::Write, S: BufRead>(label: Option<String>, delim: String, outbuf: &mut T, inbuf: S) {
    for line in inbuf.lines() {
        if let Ok(line) = line {
            let transformed = transform_text(&delim, &label, &line);
            write!(outbuf, "{}", transformed).expect("Couldn't write line to output buffer");
        } else {
            error!("Couldn't read line");
        }
    }
}

// fn transform_standard_inner(reader: )

/// Eager implementation of phrase transform - as long as the the deque contains a phrase, keep trying to add more tokens
/// Example:
///  In:  'Please, use the fax machine to send it.'
///  Out: 'Please, use the fax_machine to send it.'
fn transform_text(delim: &String, label: &Option<String>, document: &String) -> String {
    let max_ngram = MAX_NGRAM.clone();
    let min_score = MIN_SCORE.clone();
    let mut result = String::new();
    let mut current_phrase: VecDeque<String> = VecDeque::with_capacity(max_ngram);
    let mut last_token_end = 0;
    for token in TOKEN_REGEX.find_iter(document) {
        let token_string = token.as_str().to_string();
        current_phrase.push_back(token_string);
        let phrase_vec: Vec<String> = current_phrase.iter().map(|s| s.to_owned()).collect();

        if !is_significant(&phrase_vec, label, &min_score) {
            // Not a phrase - we need to all text between the last write and the current token start
            // to the result string.
            while current_phrase.len() > 1 {
                result.push_str(&current_phrase.pop_front().unwrap());
                if current_phrase.len() > 1 {
                    result.push_str(delim);
                }
            }
            result.push_str(&document[last_token_end..token.start()]);
        } else if current_phrase.len() == max_ngram {
            // Queue is full, need to flush this phrase
            // result.push_str(current_phrase.join(delim));
            while !current_phrase.is_empty() {
                result.push_str(&current_phrase.pop_front().unwrap());
                if !current_phrase.is_empty() {
                    result.push_str(delim);
                }
            }
        }
        last_token_end = token.end();
    }
    while current_phrase.len() > 0 {
        result.push_str(&current_phrase.pop_front().unwrap());
        if !current_phrase.is_empty() {
            result.push_str(delim);
        }
    }
    result.push_str(&document[last_token_end..]);
    result
}

fn is_significant(ngram: &Vec<String>, label: &Option<String>, min_score: &f64) -> bool {
    if let Some(scores) = SCORES.get(label) {
        if let Some(score) = scores.get(ngram) {
            return score > &min_score;
        }
    }
    false
}

fn list_phrase_labels() -> std::io::Result<Vec<Option<String>>> {
    let label_regex = Regex::new(r"counts_label=(.*).txt").unwrap();
    let mut labels = vec!();
    for fpath in std::fs::read_dir("data/")?.filter_map(|fpath| fpath.ok()) {
        if let Some(fname) = fpath.path().file_name() {
            let fname = fname.to_string_lossy().into_owned();
            if let Some(re_match) = label_regex.captures_iter(&fname).next() {
                labels.push(Some(re_match[1].to_string()));
            }
        }
    }
    if std::path::Path::new("data/counts_root.txt").exists() {
        labels.push(None);
    }
    Ok(labels)
}

fn list_score_labels() -> std::io::Result<Vec<Option<String>>> {
    let label_regex = Regex::new(r"scores_label=(.*).txt").unwrap();
    let mut labels = vec!();
    for fpath in std::fs::read_dir("data/")?.filter_map(|fpath| fpath.ok()) {
        if let Some(fname) = fpath.path().file_name() {
            let fname = fname.to_string_lossy().into_owned();
            if let Some(re_match) = label_regex.captures_iter(&fname).next() {
                labels.push(Some(re_match[1].to_string()));
            }
        }
    }
    if std::path::Path::new("data/scores_root.txt").exists() {
        labels.push(None);
    }
    Ok(labels)
}

fn cmd_export() {
    // Cases: 
    //  - No phrases learned - error
    //  - One label learned -  can only export phrases?
    //  - Multiple labels learned - can export phrases and unigrams
    let available_labels = list_phrase_labels().expect("Couldn't read labels from file system.");
    if available_labels.len() == 0 {
        error!("No documents counted yet, thus scores cannot be exported.");
        std::process::exit(1);
    }

    let mut label_ngrams = HashMap::new();
    read_partition_counts_for_labels(&Some(available_labels.to_owned()), &mut label_ngrams)
        .expect("Couldn't read token counts.");

    // We can export unigrams and phrases
    let label_phrase_scores = score_phrases(&label_ngrams);
    let label_ngram_scores = score_ngrams(&label_ngrams);

    map_merged_scores(&label_phrase_scores, &label_ngram_scores, |label, scores| {
        write_scores(&scores, &label);
    });
}

fn ngram_valid(ngram: &Vec<String>) -> bool {
    if ngram.first().is_none() || ngram.last().is_none() {
        false
    } else if HEAD_UNIGRAM_IGNORES.contains(&ngram.first().expect("Should have left unigram").to_owned()) || TAIL_UNIGRAM_IGNORES.contains(&ngram.last().expect("Should have right unigram").to_owned()) {
        false
    } else {
        true
    }
}

fn score_phrases(label_ngrams: &HashMap<Option<String>, HashMap<Vec<String>, i64>>) -> HashMap<Option<String>, HashMap<Vec<String>, f64>> {
    let mut label_scores = HashMap::new();
    let min_score = MIN_SCORE.clone();
    for (label, ngrams) in label_ngrams {
        let mut skipped = 0;
        let mut scores: HashMap<Vec<String>, f64> = HashMap::new();
        let total_count: i64 = ngrams.iter()
            .filter(|(ngram, _count)| ngram.len() == 1)
            .map(|(_ngram, count)| count.to_owned())
            .sum();
        let total_count: f64 = total_count as f64;
        for (ngram, ngram_count) in ngrams {
            if ngram.len() == 1 || !ngram_valid(ngram) {
                continue;
            }

            if let Some(score) = npmi_phrase_score(&total_count, ngram, ngram_count, &ngrams, &min_score) {
                scores.insert(ngram.clone(), score);
            } else {
                skipped += 1;
            }
        }
        debug!("Calculated scores for {} phrases, skipped {}.", scores.len(), skipped);
        label_scores.insert(label.to_owned(), scores);
    }
    label_scores
}

fn score_ngrams(label_ngrams: &HashMap<Option<String>, HashMap<Vec<String>, i64>>) -> HashMap<Option<String>, HashMap<Vec<String>, f64>> {
    let mut label_scores = HashMap::new();
    let min_score = MIN_SCORE.clone();
    let label_ngram_totals: HashMap<Option<String>, f64> = label_ngrams.iter().map(|(label, ngrams)| {
        (
            label.to_owned(),
            ngrams.iter()
                .filter(|(ngram, _count)| ngram.len() == 1)
                .map(|(_ngram, count)| count.to_owned() as f64)
                .sum()
        )
    }).collect();
    let all_tokens_total_count: f64 = label_ngram_totals.values().sum();
    debug!("Merging ngram counts");
    let mut ngram_counts_across_labels: HashMap<Vec<String>, i64> = HashMap::new();
    for (_label, ngrams) in label_ngrams {
        for (ngram, count) in ngrams {
            if ngram_valid(ngram) {
                if ngram_counts_across_labels.contains_key(ngram) {
                    let ngram_count = ngram_counts_across_labels.get_mut(ngram).unwrap();
                    *ngram_count += count;
                } else {
                    ngram_counts_across_labels.insert(ngram.clone(), count.clone());
                }
            }
        }
    }
    debug!("Calculating label/token npmi");
    for (label, ngrams) in label_ngrams {
        let mut skipped = 0;
        let mut scores: HashMap<Vec<String>, f64> = HashMap::new();
        let this_label_total_count = label_ngram_totals.get(label).unwrap();
        for (ngram, ngram_count) in ngrams {
            if ngram_valid(ngram) {
                let ngram_count_across_labels = ngram_counts_across_labels.get(ngram).unwrap_or(&0i64);
                if let Some(score) = npmi_label_score(&all_tokens_total_count, &this_label_total_count, &ngram_count_across_labels, ngram_count, &min_score) {
                    scores.insert(ngram.clone(), score);
                } else {
                    skipped += 1;
                }
            } else {
                skipped += 1;
            }
        }
        debug!("Calculated scores for {} ngrams, skipped {}.", scores.len(), skipped);
        label_scores.insert(label.to_owned(), scores);
    }
    debug!("Calculated label/token npmi for {} labels", label_scores.len());
    label_scores
}

fn map_merged_scores<Callback: Fn(&Option<String>, &HashMap<Vec<String>, f64>)>(label_phrase_scores: &HashMap<Option<String>, HashMap<Vec<String>, f64>>, label_ngram_scores: &HashMap<Option<String>, HashMap<Vec<String>, f64>>, f: Callback) {
    for (label, ngram_scores) in label_ngram_scores {
        debug!("Merging scores for label {:?}", label);
        if let Some(phrase_scores) = label_phrase_scores.get(&label) {
            let mut token_merged = HashMap::new();
            for (ngram, ngram_score) in ngram_scores {
                if let Some(phrase_score) = phrase_scores.get(ngram) {
                    token_merged.insert(ngram.clone(), (ngram_score * phrase_score).sqrt());
                } else {
                    token_merged.insert(ngram.clone(), ngram_score.clone());
                }
            }
            f(&label, &token_merged);
        }
    }
}

/// Perform token-wise geometric mean across labels.
fn merge_scores(label_phrase_scores: &HashMap<Option<String>, HashMap<Vec<String>, f64>>, label_ngram_scores: &HashMap<Option<String>, HashMap<Vec<String>, f64>>) -> HashMap<Option<String>, HashMap<Vec<String>, f64>> {
    let mut label_merged = HashMap::new();

    for (label, ngram_scores) in label_ngram_scores {
        debug!("Merging scores for label {:?}", label);
        if let Some(phrase_scores) = label_phrase_scores.get(&label) {
            let mut token_merged = HashMap::new();
            for (ngram, ngram_score) in ngram_scores {
                if let Some(phrase_score) = phrase_scores.get(ngram) {
                    token_merged.insert(ngram.clone(), (ngram_score * phrase_score).sqrt());
                } else {
                    token_merged.insert(ngram.clone(), ngram_score.clone());
                }
            }
            label_merged.insert(label.to_owned(), token_merged);
        }
    }

    label_merged
}

fn write_scores(scores: &HashMap<Vec<String>, f64>, label: &Option<String>) {
    let max_export = MAX_EXPORT.clone();
    let mut scores: Vec<(&Vec<String>, &f64)> = scores.iter().filter(|(_ngram, score)| {
        score.is_finite()
    }).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let mut written = 0;

    let path = match label {
        Some(label) => format!("data/scores_label={}.txt", label),
        None => "data/scores_root.txt".to_string(),
    };

    let file = File::create(path).unwrap();
    let mut file = LineWriter::new(file);
    for (ngram, score) in scores {
        let phrase = ngram.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(" "); 
        writeln!(file, "{}{}{}", phrase, FILE_DELIM, score).unwrap();

        written += 1;
        if written > max_export {
            return;
        }
    }
    file.flush().unwrap();
}

/// Calculates normalized mutual information between ngram and label
/// p_joint = ngram_count / this_label_total_count
/// pt = this_token_total_count / all_tokens_total_count
/// pl = this_label_total_count / all_labels_total_count
fn npmi_label_score(all_tokens_total_count: &f64, this_label_total_count: &f64, ngram_count_across_labels: &i64, ngram_count: &i64, min_score: &f64) -> Option<f64> {
    let pj = ngram_count.clone() as f64 / all_tokens_total_count;
    let pt = ngram_count_across_labels.clone() as f64 / all_tokens_total_count;
    let pl = this_label_total_count / all_tokens_total_count;

    if pt > 0f64 && pl > 0f64 && ngram_count >= &MIN_COUNT {
        let score = (pj / pt / pl).ln() / -pj.ln();
        if &score > min_score { Some(score) } else { None }
    } else {
        None
    }
}

fn npmi_phrase_score(corpus_size: &f64, ngram: &Vec<String>, ngram_count: &i64, ngrams: &HashMap<Vec<String>, i64>, min_score: &f64) -> Option<f64> {
    if ngram_count > &MIN_COUNT {
        let left_subgram: Vec<String> = ngram[..ngram.len() - 1].to_owned();
        let left_unigram: Vec<String> = ngram[..1].to_owned();
        let right_subgram: Vec<String> = ngram[1..].to_owned();
        let right_unigram: Vec<String> = ngram[ngram.len() - 1 ..].to_owned();

        let pj: f64 = ngram_count.clone() as f64 / corpus_size;
        let pau: f64 = ngrams.get(&left_unigram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pas: f64 = ngrams.get(&right_subgram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pa = (pas * pau).sqrt();
        let pbs: f64 = ngrams.get(&left_subgram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pbu: f64 = ngrams.get(&right_unigram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pb = (pbs * pbu).sqrt();
        let score: f64 = (pj / pa / pb).ln() / -pj.ln();

        if &score > min_score { Some(score) } else { None }
    } else {
        None
    }
}

fn label_valid(label: &Option<&String>) -> bool {
    let label_regex = Regex::new("^[a-zA-Z0-9_-]+$").unwrap();
    if let Some(label) = label {
        label_regex.is_match(label)
    } else {
        return true;
    }
}

fn assert_label_valid(label: &Option<&String>) {
    if !label_valid(label) {
        error!("Invalid label: `{}`", label.unwrap());
        std::process::exit(1);
    }
}

fn main() {
    let matches = clap_app!(phrase => 
        (version: "0.2.3")
        (author: "Stuart Axelbrooke <stuart@axelbrooke.com>")
        (about: "Detect phrases in free text data.")
        (setting: clap::AppSettings::ArgRequiredElseHelp)
        (@subcommand count =>
            (about: "Count ngrams in provided input text data")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value "Label to apply to the provided documents")
            (@arg csv: --csv "Parse input as CSV, use `label` column for label, `text` column to learn phrases")
            (setting: clap::AppSettings::ArgRequiredElseHelp)
        )
        (@subcommand serve =>
            (about: "Start API server")
            (@arg port: -p --port +takes_value "Port to listen on (default 6220)")
            (@arg host: --host +takes_value "Host to listen on (default localhost)")
        )
        (@subcommand export =>
            (about: "Export a model from the ngram counts for a given label")
            (@arg label: "The label for which to export a phrase model")
        )
        (@subcommand transform =>
            (about: "Replace detected phrases with delimiter-joiend version: fax machine -> fax_machine")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value "Label specifying which model to use for transform")
            (@arg delim: --delim +takes_value "The delimiter to use between tokens (default is _)")
            (@arg csv: --csv "Parse input as CSV, use `label` column for label, `text` column to learn phrases")
            (@arg output: -o --output +takes_value "Where to write the results")
            (setting: clap::AppSettings::ArgRequiredElseHelp)
        )
    ).get_matches();

    if let Some(matches) = matches.subcommand_matches("serve") {
        let port = matches.value_of("port").unwrap_or("6220").parse::<u16>().expect("Couldn't parse port.");
        let host = matches.value_of("host").unwrap_or("localhost");
        serve(host, port);
    } else if let Some(matches) = matches.subcommand_matches("count") {
        env_logger::init();
        let is_csv = matches.is_present("csv");
        let label = matches.value_of("label").map(|s| s.to_string());
        assert_label_valid(&label.as_ref());
        match matches.value_of("input") {
            Some(path) => {
                cmd_count(path, label, is_csv);
            },
            None => {
                error!("Must provide a file to read text from, or pass - and stream to stdin.");
                std::process::exit(1);
            }
        }
    } else if let Some(_matches) = matches.subcommand_matches("export") {
        env_logger::init();
        cmd_export();
    } else if let Some(matches) = matches.subcommand_matches("transform") {
        env_logger::init();
        let is_csv = matches.is_present("csv");
        let label = matches.value_of("label").map(|s| s.to_string());
        let delim = matches.value_of("delim").map(|s| s.to_string()).unwrap_or("_".to_string());
        let output = matches.value_of("output").map(|s| s.to_string());
        assert_label_valid(&label.as_ref());
        match matches.value_of("input") {
            Some(input) => {
                cmd_transform(input.to_string(), is_csv, label, delim, output);
            },
            None => {
                error!("Must provide a file to read text from, or pass - and stream to stdin.");
                std::process::exit(1);
            }
        }
    }
}
