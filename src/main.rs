#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;
#[macro_use] extern crate rocket_contrib;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
#[macro_use] extern crate log;
#[macro_use] extern crate lazy_static;

extern crate env_logger;
extern crate rust_stemmers; // see https://crates.io/crates/rust-stemmers

use std::io::{BufReader, BufRead};
use regex::{Regex, Match};
use std::collections::{HashMap, VecDeque, HashSet};
use std::path::Path;
use std::fs::File;
use rust_stemmers::{Algorithm, Stemmer};
use rocket_contrib::json::{Json, JsonValue};
use arrayvec::ArrayString;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

// URL regex
// [-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)

const NGRAM_MAX_CHARS: usize = 32;
const LABEL_DELIM: char = ';';

type StemmedNGram = NGram;
type NGram = ArrayString<[u8;NGRAM_MAX_CHARS]>;
type NGramHash = u64;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NGramCount {
    ngram: NGram,
    count: i64
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ScoredNGram {
    ngram: NGram,
    score: f64
}

// Maps from stemmed ngram to a list of raw ngrams and their counts
type NGramCounts = HashMap<NGram, i64>;
type CanonicalizedNGramCounts = HashMap<NGramHash, NGramCount>;
// Maps from stemmed ngram to cannonical ngram and its score
type NGramScores = HashMap<NGramHash, ScoredNGram>;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NGramCountRow {
    ngram: String,
    count: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NGramScoreRow {
    hash: NGramHash,
    ngram: String,
    score: f64,
}


#[derive(Deserialize, Serialize, Debug, Clone)]
struct Document {
    labels: Option<Vec<String>>,
    text: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct TransformDocument {
    label: Option<String>,
    text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AnalyzedDocument {
    labels: Vec<Option<String>>,
    text: String,
    ngrams: Vec<String>,
}

#[derive(Deserialize, Debug, Clone)]
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
    static ref MAX_NGRAM: usize = parse_env("MAX_NGRAM", 5);
    static ref MIN_NGRAM: usize = parse_env("MIN_NGRAM", 1);
    static ref MIN_COUNT: i64 = parse_env("MIN_COUNT", 5);
    static ref MIN_SCORE: f64 = parse_env("MIN_SCORE", 0.1f64);
    static ref MAX_EXPORT: u32 = parse_env("MAX_EXPORT", 250_000);
    static ref NGRAM_DELIM: String = std::env::var("NGRAM_DELIM").unwrap_or(" ".to_string());

    static ref HEAD_UNIGRAM_IGNORES: HashSet<String> = std::env::var("HEAD_IGNORES")
        .unwrap_or("the,a,is,and,of,to".to_string())
        .split(",").map(|s| s.to_string()).collect();

    static ref TAIL_UNIGRAM_IGNORES: HashSet<String> = std::env::var("TAIL_IGNORES")
        .unwrap_or("the,a,i,is,you,and,my,so,for".to_string())
        .split(",").map(|s| s.to_string()).collect();

    static ref STEMMER: Stemmer = {
        let algorithm: Algorithm = match &std::env::var("LANG").unwrap_or("en".to_string())[..2] {
            "ar" => Algorithm::Arabic,
            "da" => Algorithm::Danish,
            "nl" => Algorithm::Dutch,
            "en" => Algorithm::English,
            "fi" => Algorithm::Finnish,
            "fr" => Algorithm::French,
            "de" => Algorithm::German,
            "el" => Algorithm::Greek,
            "hi" => Algorithm::Hungarian,
            "it" => Algorithm::Italian,
            "pt" => Algorithm::Portuguese,
            "ro" => Algorithm::Romanian,
            "ru" => Algorithm::Russian,
            "es" => Algorithm::Spanish,
            "sv" => Algorithm::Swedish,
            "ta" => Algorithm::Tamil,
            "tk" => Algorithm::Turkish,
            _ => panic!(r#"Invalid stemming language, please choose one of the following:
    Arabic
    Danish
    Dutch
    English
    Finnish
    French
    German
    Greek
    Hungarian
    Italian
    Portuguese
    Romanian
    Russian
    Spanish
    Swedish
    Tamil
    Turkish
"#),
    };
        Stemmer::create(algorithm)
    };
    static ref TOKEN_REGEX: Regex = Regex::new(&std::env::var("TOKEN_REGEX").unwrap_or(r"[\w+'’]+".to_string())).unwrap();
    static ref CHUNK_SPLIT_REGEX: Regex = Regex::new(&std::env::var("CHUNK_SPLIT_REGEX").unwrap_or(r"[\.\?!\(\);]+".to_string())).unwrap();
}

lazy_static! {
    static ref SCORES: HashMap<Option<String>, NGramScores> = {
        let mut label_ngrams = HashMap::new();
        read_partition_scores_for_labels(&Some(list_score_labels().unwrap()), &mut label_ngrams).expect("Unable to read partitions for labels");
        label_ngrams
    };
}

fn ngram_from_str(s: &String) -> Option<NGram> {
    if s.len() < NGRAM_MAX_CHARS {
        Some(NGram::from(s).expect("Couldn't build ngram from string"))
    } else {
        None
    }
}

fn ngram_to_str(ngram: &NGram) -> String {
    ngram.to_string()
}

/// Hashes a token vector.  Doesn't do stemming for you.
fn ngram_hash_vec(token_vec: &Vec<String>) -> NGramHash {
    let mut hasher = DefaultHasher::new();
    for token in token_vec.iter() {
        hasher.write(token.as_bytes());
    }
    hasher.finish()
}

fn hash_ngram(ngram: &NGram) -> NGramHash {
    let ngram_delim = NGRAM_DELIM.clone();
    ngram_hash_vec(&ngram.split(&ngram_delim).map(|s| STEMMER.stem(&s.to_lowercase()).to_string()).collect())
}

fn analyze_text(text: &String, scores: &NGramScores, max_ngram: &usize, min_score: &f64) -> Vec<String> {
    let mut significant_ngrams: Vec<NGram> = vec!();

    for chunk in CHUNK_SPLIT_REGEX.split(&text) {
        let mut token_queues: Vec<VecDeque<String>> = Vec::new();
        for i in 1..max_ngram+1 {
            token_queues.push(VecDeque::with_capacity(i));
        }
        for token in TOKEN_REGEX.find_iter(chunk) {
            let token_string = re_match_stem(token);
            for i in 1..max_ngram+1 {
                if let Some(queue) = token_queues.get_mut(i - 1) {
                    queue.push_back(token_string.to_owned());
                    if queue.len() > i {
                        queue.pop_front();
                    }
                    if queue.len() == i {
                        let ngram_hash = ngram_hash_vec(&queue.iter().map(|s| s.to_string()).collect());
                        if let Some(scored_ngram) = scores.get(&ngram_hash) {
                            if &scored_ngram.score > min_score {
                                significant_ngrams.push(scored_ngram.ngram);
                            }
                        }
                    }
                }
            }
        }
    }

    significant_ngrams.sort();
    significant_ngrams.dedup();

    significant_ngrams.iter().map(|s| s.to_string()).collect()
}

#[post("/analyze", data = "<data>")]
fn api_analyze(data: Json<ApiAnalyzeRequest>) -> JsonValue {
    let mut documents = data.0.documents.clone();
    let max_ngram = MAX_NGRAM.clone();
    let min_score = MIN_SCORE.clone();
    let analyzed_docs: Vec<AnalyzedDocument> = documents.iter_mut().map(|d| {
        let mut labels: Vec<Option<String>> = vec!();
        if let Some(doc_labels) = &d.labels {
            for label in doc_labels {
                labels.push(Some(label.to_owned()));
            }
        } else {
            labels = vec!(None);
        }
        
        let mut significant_terms = vec!();
        for label in &labels {
            if let Some(scores) = SCORES.get(label) {
                significant_terms.extend(analyze_text(&d.text, scores, &max_ngram, &min_score));
            }
        }
        AnalyzedDocument {
            labels: labels,
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

fn re_match_stem(re_match: Match) -> String {
    STEMMER.stem(&re_match.as_str().replace('’', "'").to_lowercase()).to_string()
}

fn read_partition_counts_for_labels(labels: &Vec<Option<String>>, label_ngrams: &mut HashMap<Option<String>, NGramCounts>) -> std::io::Result<()> {
    for label in labels {
        let ngrams = match read_partition_counts(&label.as_ref()) {
            Ok(Some(ngrams)) => ngrams,
            Ok(None) => HashMap::new(),
            Err(err) => return Err(err),
        };
        label_ngrams.insert(label.to_owned(), ngrams);
    }
    Ok(())
}

fn read_partition_scores_for_labels(labels: &Option<Vec<Option<String>>>, label_scores: &mut HashMap<Option<String>, NGramScores>) -> std::io::Result<()> {
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

fn update_phrase_model(label: Option<String>, documents: &mut Vec<String>) -> std::io::Result<()> {
    const BATCH_SIZE: usize = 100_000;
    documents.sort();
    documents.dedup();

    for batch in &documents.into_iter().chunks(BATCH_SIZE) {
        let mut ngrams = read_partition_counts(&label.as_ref())?.unwrap_or(NGramCounts::new());
        let docs: Vec<String> = batch.map(|s| s.to_owned()).collect();
        let new_ngrams = count_ngrams(&docs);
        if ngrams.len() > 0 {
            merge_ngrams_into(&new_ngrams, &mut ngrams);
            write_partition_counts(&label, &ngrams)?;
        } else {
            write_partition_counts(&label, &new_ngrams)?;
        }
    }
    
    Ok(())
}

fn update_phrase_models_from_labeled_documents(labeled_documents: &mut Vec<LabeledDocument>) -> std::io::Result<()> {
    let label_delim = LABEL_DELIM.clone();
    let mut groups: HashMap<Option<String>, Vec<String>> = HashMap::new();
    for labeled_document in labeled_documents.iter() {
        let labels: Vec<Option<String>> = match &labeled_document.label {
            None => vec!(None),
            Some(label) => {
                if label == "" {
                    vec!(None)
                } else {
                    label.split(label_delim).map(|s| Some(s.to_string())).collect()
                }
            },
        };
        for label in labels {
            if let Some(group) = groups.get_mut(&label) {
                group.push(labeled_document.text.clone());
            } else {
                assert_label_valid(&label.to_owned().as_ref());
                groups.insert(label.clone(), vec!(labeled_document.text.clone()));
            }
        }
    }

    debug!("Counting ngrams for labels: {:?}", groups.keys());

    groups.par_iter_mut().for_each(move |(label, mut documents)| {
            update_phrase_model(label.clone(), &mut documents).expect("Thread failed in phrase model update");
    });

    Ok(())
}

fn merge_ngrams_into(from: &NGramCounts, into: &mut NGramCounts) {
    debug!("Merging {} ngrams into {} ngrams.", from.len(), into.len());
    for (ngram, count) in from {
        if let Some(into_count) = into.get_mut(ngram) {
            *into_count += count;
        } else {
            into.insert(ngram.clone(), count.clone());
        }
    }
}

fn merge_ngrams_into_owned(from: NGramCounts, into: &mut NGramCounts) {
    for (ngram, count) in from {
        if let Some(into_count) = into.get_mut(&ngram) {
            *into_count += count;
        } else {
            into.insert(ngram, count);
        }
    }
}

fn count_ngrams_into(documents: &Vec<String>, ngrams: &mut NGramCounts) {
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

fn count_ngrams(documents: &Vec<String>) -> NGramCounts {
    let mut ngrams = NGramCounts::new();
    debug!("Counting ngrams for {} documents.", documents.len());
    count_ngrams_into(documents, &mut ngrams);
    ngrams
}

fn count_document_ngrams(document: &String, ngrams: &mut NGramCounts, max_ngram: &usize) {
    let mut doc_ngrams = NGramCounts::new();
    let delim = NGRAM_DELIM.clone();

    for chunk in CHUNK_SPLIT_REGEX.split(&document) {
        let mut token_queues: Vec<VecDeque<String>> = Vec::new();
        for i in 1..max_ngram+1 {
            token_queues.push(VecDeque::with_capacity(i));
        }

        for token in TOKEN_REGEX.find_iter(chunk) {
            let token_string: String = token.as_str().to_string();

            for i in 1..max_ngram+1 {
                if let Some(token_queue) = token_queues.get_mut(i - 1) {
                    
                    token_queue.push_back(token_string.clone());
                    if token_queue.len() > i {
                        token_queue.pop_front();
                    }

                    if token_queue.len() == i {
                        let token_queue: Vec<String> = token_queue.iter().map(|s| s.to_string()).collect();
                        let string = token_queue.join(&delim);
                        if string.len() < NGRAM_MAX_CHARS {
                            let string = NGram::from(&string).expect("Couldn't build ArrayString from phrase");
                            doc_ngrams.insert(string, 1);
                        }
                    }
                }
            }
        }
    }

    merge_ngrams_into_owned(doc_ngrams, ngrams);
}

fn prune_ngrams(ngrams: &mut NGramCounts) {
    let ngrams_len = ngrams.len();
    if ngrams_len > PRUNE_AT.clone() {
        debug!("Pruning ngrams of length {}.", ngrams_len);

        let mut ngram_pairs: Vec<(&NGram, &i64)> = ngrams.iter().collect();
        ngram_pairs.sort_by(|a, b| b.1.cmp(&a.1));

        let stems_to_prune: Vec<NGram> = ngram_pairs.iter().skip(PRUNE_TO.clone()).map(|(phrase, _count)| phrase.to_owned().to_owned()).collect();
        for stem in stems_to_prune.iter() {
            ngrams.remove(stem);
        }
        debug!("Pruned to {} ngrams.", ngrams.len());
    }
}

fn read_partition_counts(label: &Option<&String>) -> std::io::Result<Option<NGramCounts>> {
    debug!("Reading counts for label={:?}", label);
    let path_str: String = match label {
        Some(label) => format!("data/counts_label={}.csv", label),
        None => String::from("data/counts_default.csv"),
    };
    let path = Path::new(&path_str);
    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        let mut ngrams = NGramCounts::new();
        for row in reader.deserialize() {
            if let Ok(ngram_count) = row {
                let ngram_count: NGramCountRow = ngram_count;
                if ngram_count.ngram.len() < NGRAM_MAX_CHARS {
                    ngrams.insert(NGram::from(&ngram_count.ngram).expect("Couldn't construct ArrayString from ngram"), ngram_count.count);
                }
            }
        }

        Ok(Some(ngrams))
    } else {
        Ok(None)
    }
}

fn read_partition_scores(label: &Option<&String>) -> std::io::Result<Option<NGramScores>> {
    let path_str: String = match label {
        Some(label) => format!("data/scores_label={}.csv", label),
        None => String::from("data/scores_default.csv"),
    };
    let path = Path::new(&path_str);
    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        let mut scores = NGramScores::new();
        for row in reader.deserialize() {
            if let Ok(row) = row {
                let ngram_score: NGramScoreRow = row;
                if let Some(ngram) = ngram_from_str(&ngram_score.ngram) {
                    scores.insert(ngram_score.hash, ScoredNGram { ngram: ngram, score: ngram_score.score });
                }
            }
        }

        Ok(Some(scores))
    } else {
        Ok(None)
    }
}

fn write_partition_counts(label: &Option<String>, ngrams: &NGramCounts) -> std::io::Result<()> {
    debug!("Writing partition {:?}.", &label);
    let path_str: String = match label {
        Some(label) => format!("data/counts_label={}.csv", label),
        None => String::from("data/counts_default.csv"),
    };
    let mut writer = csv::Writer::from_path(path_str)?;

    let mut rows: Vec<NGramCountRow> = vec!();

    for (phrase, count) in ngrams {
        if count > &1 {
            rows.push(NGramCountRow { ngram: phrase.to_string(), count: *count});
        }
    }

    rows.sort_by_key(|row| -row.count);

    for row in rows.iter().take(PRUNE_TO.clone()) {
        writer.serialize(row)?;
    }

    Ok(())
}

fn count_stdin(labels: Option<Vec<String>>, is_csv: bool) {
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
        if let Some(labels) = labels {
            for label in labels.iter() {
                update_phrase_model(Some(label.clone()), &mut documents).expect("Failed to update phrase models");
            }
        } else {
            update_phrase_model(None, &mut documents).expect("Failed to update phrase models");
        }
    }
}

#[derive(Deserialize)]
struct LabeledDocument {
    label: Option<String>,
    text: String,
}

fn count_file(path: &str, labels: Option<Vec<String>>, is_csv: bool) {
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
        if let Some(labels) = labels {
            for label in labels.iter() {
                update_phrase_model(Some(label.clone()), &mut documents).expect("Failed to update phrase models");
            }
        } else {
            update_phrase_model(None, &mut documents).expect("Failed to update phrase models");
        }
    };
}

fn cmd_count(path: &str, labels: Option<Vec<String>>, is_csv: bool) {
    if is_csv && labels.is_some() {
        error!("Cannot specify label and provide a CSV");
        std::process::exit(1);
    }
    std::fs::create_dir_all("data").expect("Failed to ensure data directory existence.");
    if path == "-" {
        count_stdin(labels, is_csv);
    } else {
        count_file(path, labels, is_csv);
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
        let document: TransformDocument = result.expect("Unable to parse csv document");
        let transformed = transform_text(delim, &document.label, &document.text);
        csv_writer.serialize(TransformDocument { label: document.label, text: transformed}).expect("Couldn't write CSV output");
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


// Allocates the given window to known phrases as best possible.  Returns number of stems consumed.
fn allocate_ngrams(stem_window: &mut Vec<String>, buf: &mut String, label: &Option<String>, min_score: &f64, delim: &String) -> usize {
    let window_len = stem_window.len();
    let ngram_delim = NGRAM_DELIM.clone();
    if window_len > 1 {
        let ngram_hash = ngram_hash_vec(stem_window);
        if let Some(scored_ngram) = is_significant(&ngram_hash, label, min_score) {
            buf.push_str(&scored_ngram.ngram.replace(&ngram_delim, delim));
            window_len
        } else {
            allocate_ngrams(&mut stem_window[..window_len - 1].to_vec(), buf, label, min_score, delim)
        }
    } else {
        0
    }
}


/// Eager implementation of phrase transform - as long as the the deque contains a phrase, keep trying to add more tokens
/// Example:
///  In:  'Please, use the fax machine to send it.'
///  Out: 'Please, use the fax_machine to send it.'
fn transform_text(delim: &String, label: &Option<String>, document: &String) -> String {
    let max_ngram = MAX_NGRAM.clone();
    let min_score = MIN_SCORE.clone();
    let mut result = String::new();
    let mut current_phrase: VecDeque<Match> = VecDeque::with_capacity(max_ngram);
    let mut last_token_end = 0;
    for token in TOKEN_REGEX.find_iter(document) {
        current_phrase.push_back(token);
        if current_phrase.len() == max_ngram {
            result.push_str(&document[last_token_end..current_phrase.get(0).expect("Should have been able to get head token").start()]);
            // At this point, the result string is up to date (apart from stems in the queue already)
            let mut stem_window: Vec<String> = current_phrase.iter().map(|m| re_match_stem(m.clone())).collect();
            let tokens_written = allocate_ngrams(&mut stem_window, &mut result, label, &min_score, delim);
            if tokens_written == 0 {
                // No tokens were written, need to write the first one
                let first_token = current_phrase.pop_front().unwrap();
                result.push_str(first_token.as_str());
                last_token_end = first_token.end();
            } else {
                // Tokens were written, we need to write the chars between the last written token and the first token still in the queue
                last_token_end = current_phrase.get(tokens_written - 1).expect("Should have been able to get the last written token").end();
                for _idx in 0..tokens_written {
                    current_phrase.pop_front();
                }
            }
        }
    }

    result.push_str(&document[last_token_end..]);
    result
}

fn is_significant(ngram_hash: &u64, label: &Option<String>, min_score: &f64) -> Option<ScoredNGram> {
    if let Some(scores) = SCORES.get(label) {
        if let Some(ngram_score) = scores.get(&ngram_hash) {
            if &ngram_score.score > min_score {
                return Some(ngram_score.clone());
            }
        }
    }
    None
}

fn list_phrase_labels() -> std::io::Result<Vec<Option<String>>> {
    let label_regex = Regex::new(r"counts_label=(.*).csv").unwrap();
    let mut labels = vec!();
    for fpath in std::fs::read_dir("data/")?.filter_map(|fpath| fpath.ok()) {
        if let Some(fname) = fpath.path().file_name() {
            let fname = fname.to_string_lossy().into_owned();
            if let Some(re_match) = label_regex.captures_iter(&fname).next() {
                labels.push(Some(re_match[1].to_string()));
            }
        }
    }
    if Path::new("data/counts_default.csv").exists() {
        labels.push(None);
    }
    Ok(labels)
}

fn list_score_labels() -> std::io::Result<Vec<Option<String>>> {
    let label_regex = Regex::new(r"scores_label=(.*).csv").unwrap();
    let mut labels = vec!();
    for fpath in std::fs::read_dir("data/")?.filter_map(|fpath| fpath.ok()) {
        if let Some(fname) = fpath.path().file_name() {
            let fname = fname.to_string_lossy().into_owned();
            if let Some(re_match) = label_regex.captures_iter(&fname).next() {
                labels.push(Some(re_match[1].to_string()));
            }
        }
    }
    if std::path::Path::new("data/scores_default.csv").exists() {
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

    let mut label_ngrams = load_canonicalized_ngrams(&available_labels.to_owned()).expect("Couldn't load ngram counts form disk");

    if label_ngrams.len() == 0 {
        warn!("Found no ngram counts, exiting.");
        std::process::exit(0);
    }

    let default_ngrams = label_ngrams.remove(&None).expect("Didn't find default ngrams in loaded ngrams.");

    for (label, ngrams) in label_ngrams {
        let scores = score_ngrams(ngrams, &default_ngrams);
        write_scores(&label.clone(), &scores);
        debug!("Finished scoring label {:?}", label);
    };

    write_scores(&None, &score_default_ngrams(default_ngrams));
}

fn ngram_valid(ngram: &NGram) -> bool {
    let delim = NGRAM_DELIM.clone();
    let lower = ngram.to_lowercase();
    let mut split = lower.split(&delim);
    let first = split.next();
    let last = split.last();
    if let Some(right_unigram) = last {
        if HEAD_UNIGRAM_IGNORES.contains(&first.expect("Should have left unigram").to_owned()) || TAIL_UNIGRAM_IGNORES.contains(&right_unigram.to_owned()) {
            return false;
        }
    }
    true
}

fn load_canonicalized_ngrams(labels: &Vec<Option<String>>) -> std::io::Result<HashMap<Option<String>, CanonicalizedNGramCounts>> {
    let mut label_ngrams = HashMap::new();
    read_partition_counts_for_labels(labels, &mut label_ngrams).expect("Couldn't read partition counts");
    Ok(canonicalize_ngrams(&mut label_ngrams))
}

fn canonicalize_ngrams(label_ngrams: &mut HashMap<Option<String>, NGramCounts>) -> HashMap<Option<String>, CanonicalizedNGramCounts> {
    // Add default partition as the sum over labels per ngram count
    let mut default_ngrams = label_ngrams.remove(&None).unwrap_or(NGramCounts::new());

    for (_label, ngrams) in label_ngrams.iter() {
        merge_ngrams_into(ngrams, &mut default_ngrams);
    }

    debug!("Grouping ngrams by stemmed hash");
    let mut ngram_counts_map: HashMap<NGramHash, Vec<NGramCount>> = HashMap::new();
    for (ngram, count) in &default_ngrams {
        let ngram_hash = hash_ngram(&ngram);
        if let Some(ngram_counts) = ngram_counts_map.get_mut(&ngram_hash) {
            ngram_counts.push(NGramCount { ngram: ngram.clone(), count: count.clone() });
        } else {
            ngram_counts_map.insert(ngram_hash, vec!(NGramCount { ngram: ngram.clone(), count: count.clone() }));
        }
    }

    debug!("Choosing top volume ngrams as canonical");
    let mut canon_ngrams = CanonicalizedNGramCounts::new();
    for (ngram_hash, ngram_counts) in ngram_counts_map {
        let count_sum = ngram_counts.iter().map(|ngc| ngc.count).sum();
        let mut canonical_ngram = ngram_counts.iter().max_by_key(|ngc| ngc.count).expect("ngram counts shouldn't be empty").clone();
        canonical_ngram.count = count_sum;
        canon_ngrams.insert(ngram_hash, canonical_ngram);
    }

    let mut canonicalized: HashMap<Option<String>, CanonicalizedNGramCounts> = HashMap::new();
    for (label, ngrams) in label_ngrams {
        debug!("Assigning canonical ngrams for label {:?}", label);
        let mut canonized_ngrams = CanonicalizedNGramCounts::new();
        for (ngram, count) in ngrams {
            let ngram_hash = hash_ngram(&ngram);
            if let Some(canon_ngram) = canonized_ngrams.get_mut(&ngram_hash) {
                canon_ngram.count += count.clone();
            } else {
                let canon_ngram = NGramCount { ngram: canon_ngrams.get(&ngram_hash).expect("Didn't find stem in canon ngrams").ngram.clone(), count: count.clone() };
                canonized_ngrams.insert(ngram_hash, canon_ngram);
            }
        }
        debug!("Inserting canonicalized ngrams of size {} for label {:?}", canonized_ngrams.len(), label);
        canonicalized.insert(label.to_owned(), canonized_ngrams);
    }

    canonicalized.insert(None, canon_ngrams);

    debug!("Done loading canonicalized ngrams");
    canonicalized
}

fn score_ngrams(ngram_counts: CanonicalizedNGramCounts, default_ngram_counts: &CanonicalizedNGramCounts) -> NGramScores {
    // Score tokens as a function of ngram_counts and default_ngram_counts
    // Score phrases as a funciton of ngram_counts alone

    let ngram_delim = NGRAM_DELIM.clone();
    let mut scores = NGramScores::new();
    let min_ngram = MIN_NGRAM.clone();

    let min_score = MIN_SCORE.clone();
    let min_count = MIN_COUNT.clone();

    let all_tokens_total_count: i64 = default_ngram_counts.iter()
                .filter(|(_ngram_hash, ngram_count)| !ngram_count.ngram.contains(&ngram_delim))
                .map(|(_ngram_hash, ngram_count)| ngram_count.count).sum();
    let all_tokens_total_count: f64 = all_tokens_total_count as f64;

    let this_label_total_count: i64 = ngram_counts.iter()
                .filter(|(_ngram_hash, ngram_count)| !ngram_count.ngram.contains(&ngram_delim))
                .map(|(_ngram_hash, ngram_count)| ngram_count.count).sum();;
    let this_label_total_count: f64 = this_label_total_count as f64;

    for (ngram_hash, ngram_count) in &ngram_counts {
        if !ngram_valid(&ngram_count.ngram) {
            continue;
        }

        // NGram intersect Label NPMI
        let ngram_count_across_labels = default_ngram_counts.get(&ngram_hash).map(|ngc| ngc.count).unwrap_or(ngram_count.count.clone());
        if let Some(mut score) = npmi_label_score(&all_tokens_total_count, &this_label_total_count, &ngram_count_across_labels, &ngram_count.count) {

            // NGram intersect Tokens NPMI
            if ngram_count.ngram.matches(&ngram_delim).count() >= min_ngram {
                let phrase_score = npmi_phrase_score(&this_label_total_count, &ngram_count, &ngram_counts, &min_count);
                if let Some(phrase_score) = phrase_score {
                    // Geometric Mean - NOTE: this means if either score is less than zero, the ngram is essentially ignored
                    // score = (score * phrase_score).sqrt();
                    // Arithmetic Mean
                    score = (score + phrase_score) / 2f64;
                }
            }
            
            if score > min_score {
                scores.insert(ngram_hash.clone(), ScoredNGram { ngram: ngram_count.ngram.clone(), score: score });
            }
        }
    }

    scores
}

fn score_default_ngrams(default_ngram_counts: CanonicalizedNGramCounts) -> NGramScores {
    let mut scores = NGramScores::new();
    let ngram_delim = NGRAM_DELIM.clone();

    let min_score = MIN_SCORE.clone();
    let min_count = MIN_COUNT.clone();
    let mut min_ngram = MIN_NGRAM.clone();
    if min_ngram == 1 {
        // Can't score default unigrams, since it has no "background" vocab
        min_ngram = 2;
    }

    let all_tokens_total_count: i64 = default_ngram_counts.iter()
                .filter(|(_ngram_hash, ngram_count)| !ngram_count.ngram.contains(&ngram_delim))
                .map(|(_ngram_hash, ngram_count)| ngram_count.count).sum();
    let all_tokens_total_count: f64 = all_tokens_total_count as f64;

    for (ngram_hash, ngram_count) in &default_ngram_counts {
        if !ngram_valid(&ngram_count.ngram) {
            continue;
        }
        
        if ngram_count.ngram.matches(&ngram_delim).count() >= min_ngram {
            let phrase_score = npmi_phrase_score(&all_tokens_total_count, &ngram_count, &default_ngram_counts, &min_count);
            if let Some(phrase_score) = phrase_score {
                if phrase_score > min_score {
                    scores.insert(ngram_hash.clone(), ScoredNGram { ngram: ngram_count.ngram.clone(), score: phrase_score });
                }
            }
        }
    }

    scores
}

fn write_scores(label: &Option<String>, scores: &NGramScores) {
    let max_export = MAX_EXPORT.clone();
    let mut scores: Vec<NGramScoreRow> = scores.iter().filter_map(|(ngram_hash, ngram_score)| {
        if ngram_score.score.is_finite() {
            Some(NGramScoreRow { ngram: ngram_to_str(&ngram_score.ngram), hash: ngram_hash.clone(), score: ngram_score.score })
        } else {
            None
        }
    }).collect();
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut written = 0;

    let path = match label {
        Some(label) => format!("data/scores_label={}.csv", label),
        None => "data/scores_default.csv".to_string(),
    };

    let mut writer = csv::Writer::from_path(path).expect("Couldn't open scores CSV for writing.");
    for row in scores {
        writer.serialize(row).expect("Couldn't write row to CSV");
        written += 1;
        if written > max_export {
            return;
        }
    }
}

/// Calculates normalized mutual information between ngram and label
/// p_joint = ngram_count / this_label_total_count
/// pt = this_token_total_count / all_tokens_total_count
/// pl = this_label_total_count / all_labels_total_count
fn npmi_label_score(all_tokens_total_count: &f64, this_label_total_count: &f64, ngram_count_across_labels: &i64, ngram_count: &i64) -> Option<f64> {
    let pj = ngram_count.clone() as f64 / all_tokens_total_count;
    let pt = ngram_count_across_labels.clone() as f64 / all_tokens_total_count;
    let pl = this_label_total_count / all_tokens_total_count;

    if pt > 0f64 && pl > 0f64 && ngram_count >= &MIN_COUNT {
        let score = (pj / pt / pl).ln() / -pj.ln();
        Some(score)
    } else {
        None
    }
}

fn npmi_phrase_score(corpus_size: &f64, ngram_count: &NGramCount, ngrams: &CanonicalizedNGramCounts, min_count: &i64) -> Option<f64> {
    let count = ngram_count.count;
    let ngram_delim = NGRAM_DELIM.clone();
    if &count > min_count {
        let token_vec: Vec<String> = ngram_count.ngram.split(&ngram_delim).map(|s| STEMMER.stem(&s.to_lowercase()).to_string()).collect();

        let left_subgram: NGramHash = ngram_hash_vec(&token_vec[..token_vec.len() - 1].to_vec());
        let left_unigram: NGramHash = ngram_hash_vec(&token_vec[..1].to_vec());
        let right_subgram: NGramHash = ngram_hash_vec(&token_vec[1..].to_vec());
        let right_unigram: NGramHash = ngram_hash_vec(&token_vec[token_vec.len() - 1 ..].to_vec());

        let pj: f64 = count.clone() as f64 / corpus_size;
        let pau: f64 = ngrams.get(&left_unigram).map(|ngc| ngc.count).unwrap_or(count).clone() as f64 / corpus_size;
        let pas: f64 = ngrams.get(&right_subgram).map(|ngc| ngc.count).unwrap_or(count).clone() as f64 / corpus_size;
        let pa = (pas * pau).sqrt();
        let pbs: f64 = ngrams.get(&left_subgram).map(|ngc| ngc.count).unwrap_or(count).clone() as f64 / corpus_size;
        let pbu: f64 = ngrams.get(&right_unigram).map(|ngc| ngc.count).unwrap_or(count).clone() as f64 / corpus_size;
        let pb = (pbs * pbu).sqrt();
        let score: f64 = (pj / pa / pb).ln() / -pj.ln();

        Some(score)
    } else {
        None
    }
}

fn show_label(label: &Option<String>, num: &usize) {
    let file = if let Some(label) = label {
        println!("\nLabel={}", label);
        File::open(format!("data/scores_label={}.csv", label)).expect("Couldn't read scores file.")
    } else {
        println!("\nDefault");
        File::open("data/scores_default.csv").expect("Couldn't read scores file.")
    };
    
    let reader = BufReader::new(file);
    for line in reader.lines().take(*num + 1) {
        println!("{}", line.expect("Couldn't read line from scores file"));
    }
}

fn cmd_show(labels: Vec<Option<String>>, num: usize) {
    for label in labels {
        show_label(&label, &num);
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
        (version: "0.3.0")
        (author: "Stuart Axelbrooke <stuart@axelbrooke.com>")
        (about: "Detect phrases in free text data.")
        (setting: clap::AppSettings::ArgRequiredElseHelp)
        (@subcommand count =>
            (about: "Count ngrams in provided input text data")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value ... "Label to apply to the provided documents")
            (@arg csv: --csv "Parse input as CSV, use `label` column for label, `text` column to learn phrases")
            (@arg workers: --workers -w +takes_value "Number of workers to use, defaults to number of system threads.")
            (setting: clap::AppSettings::ArgRequiredElseHelp)
        )
        (@subcommand serve =>
            (about: "Start API server")
            (@arg port: -p --port +takes_value "Port to listen on (default 6220)")
            (@arg host: --host +takes_value "Host to listen on (default localhost)")
        )
        (@subcommand export =>
            (about: "Export a model from the ngram counts for a given label")
            (@arg workers: --workers -w +takes_value "Number of workers to use, defaults to number of system threads.")
            (@arg label: "The label for which to export a phrase model")
        )
        (@subcommand transform =>
            (about: "Replace detected phrases with delimiter-joiend version: fax machine -> fax_machine")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value "Label specifying which model to use for transform")
            (@arg delim: --delim +takes_value "The delimiter to use between tokens (default is _)")
            (@arg csv: --csv "Parse input as CSV, use `label` column for label, `text` column to learn phrases")
            (@arg output: -o --output +takes_value "Where to write the results")
        )
        (@subcommand show =>
            (about: "Show top scoring ngrams per label")
            (@arg label: -l --label +takes_value ... "Label specifying which model to use for transform")
            (@arg num: -n --num +takes_value "Top N-scored phrases to print")
        )
    ).get_matches();

    if let Some(matches) = matches.subcommand_matches("serve") {
        let port = matches.value_of("port").unwrap_or("6220").parse::<u16>().expect("Couldn't parse port.");
        let host = matches.value_of("host").unwrap_or("localhost");
        serve(host, port);
    } else if let Some(matches) = matches.subcommand_matches("count") {
        env_logger::init();
        let is_csv = matches.is_present("csv");
        let labels: Option<Vec<String>> = matches.values_of("label").map(|v| v.map(|s| s.to_string()).collect());
        if let Some(num_workers) = matches.value_of("workers").map(|s| s.parse::<usize>().expect("Couldn't parse --workers")) {
            std::env::set_var("RAYON_NUM_THREADS", num_workers.to_string());
        }
        if let Some(labels) = &labels {
            for label in labels.iter() {
                assert_label_valid(&Some(label));
            }
        }
        match matches.value_of("input") {
            Some(path) => {
                cmd_count(path, labels, is_csv);
            },
            None => {
                error!("Must provide a file to read text from, or pass - and stream to stdin.");
                std::process::exit(1);
            }
        }
    } else if let Some(_matches) = matches.subcommand_matches("export") {
        env_logger::init();
        if let Some(num_workers) = matches.value_of("workers").map(|s| s.parse::<usize>().expect("Couldn't parse --workers")) {
            std::env::set_var("RAYON_NUM_THREADS", num_workers.to_string());
        }
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
    } else if let Some(matches) = matches.subcommand_matches("show") {
        let mut labels: Vec<Option<String>> = vec!();
        if let Some(found_labels) = matches.values_of("label") {
            labels.extend(found_labels.map(|l| Some(l.to_string())));
        } else {
            labels.extend(list_score_labels().expect("Couldn't list labels"));
        }
        let num: usize = matches.value_of("num").map(|s| s.parse::<usize>().expect("Couldn't parse --num")).unwrap_or(5);
        cmd_show(labels, num);
    }
}
