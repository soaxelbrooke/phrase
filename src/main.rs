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
    labels: Option<Vec<Option<String>>>,
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

#[derive(Deserialize, Debug, Clone)]
struct ApiTransformRequest {
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

#[derive(PartialEq, Debug, Clone)]
enum ParseMode {
    CSV,
    JSON,
    PlainText
}

impl std::str::FromStr for ParseMode {
    type Err = ();
    fn from_str(s: &str) -> Result<ParseMode, ()> {
        match &s.to_lowercase()[..] {
            "csv" => Ok(ParseMode::CSV),
            "json" => Ok(ParseMode::JSON),
            "plain" => Ok(ParseMode::PlainText),
            "plaintext" => Ok(ParseMode::PlainText),
            _ => Err(()),
        }
    }
}

enum Input {
    Standard(std::io::Stdin),
    File(std::fs::File),
}

impl Input {
    fn stdin() -> Input {
        Input::Standard(std::io::stdin())
    }

    fn file(path: String) -> std::io::Result<Input> {
        Ok(Input::File(std::fs::File::open(path)?))
    }

    fn from_arg(arg: Option<String>) -> std::io::Result<Input> {
        Ok(match arg {
            None       => Input::stdin(),
            Some(path) => if path == "-" { Input::stdin() } else { Input::file(path)? },
        })
    }
}

impl std::io::Read for Input {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match *self {
            Input::Standard(ref mut s) => s.read(buf),
            Input::File(ref mut f)     => f.read(buf),
        }
    }
}

enum ReaderStrategy {
    CSV(csv::Reader<Input>),
    File(BufReader<Input>),
}

struct BatchedInputReader {
    reader: ReaderStrategy,
    batch_size: u64,
    mode: ParseMode,
    text_fields: Vec<String>,
    label_fields: Vec<String>,
    labels: Vec<Option<String>>,
}

impl BatchedInputReader {
    fn new(input: Input, mode: ParseMode, batch_size: u64, text_fields: Vec<String>, label_fields: Vec<String>, labels: Vec<Option<String>>) -> BatchedInputReader {
        if mode == ParseMode::CSV {
            BatchedInputReader {
                reader: ReaderStrategy::CSV(csv::Reader::from_reader(input)),
                mode: mode,
                batch_size: batch_size,
                text_fields: text_fields,
                label_fields: label_fields,
                labels: labels,
            }
        } else {
            BatchedInputReader {
                reader: ReaderStrategy::File(BufReader::new(input)),
                mode: mode,
                batch_size: batch_size,
                text_fields: text_fields,
                label_fields: label_fields,
                labels: labels,
            }
        }
    }

    fn read_batch(&mut self) -> Option<Vec<Document>> {
        let documents = match &mut self.reader {
            ReaderStrategy::CSV(ref mut reader) => read_documents_from_csv(reader, &self.text_fields, &self.label_fields, &self.batch_size).ok(),
            ReaderStrategy::File(ref mut reader) => match &self.mode {
                ParseMode::JSON => read_documents_from_json(reader, &self.text_fields,&self.label_fields, &self.batch_size).ok(),
                ParseMode::PlainText => read_documents_from_plain(reader, self.labels.clone(), &self.batch_size).ok(),
                _ => None
            }
        };
        match documents {
            Some(vec) => if vec.is_empty() { None } else { Some(vec) },
            None => None,
        }
    }
}

impl Iterator for BatchedInputReader {
    type Item = Vec<Document>;

    fn next(&mut self) -> Option<Vec<Document>> {
        self.read_batch()
    }
}

enum Output {
    Standard(std::io::Stdout),
    File(std::fs::File),
}

impl Output {
    fn stdout() -> Output {
        Output::Standard(std::io::stdout())
    }

    fn file(path: String) -> std::io::Result<Output> {
        Ok(Output::File(std::fs::File::create(path)?))
    }

    fn from_arg(arg: Option<String>) -> std::io::Result<Output> {
        Ok(match arg {
            None       => Output::stdout(),
            Some(path) => if path == "-" { Output::stdout() } else { Output::file(path)? },
        })
    }
}

impl std::io::Write for Output {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match *self {
            Output::Standard(ref mut s) => s.write(buf),
            Output::File(ref mut f)     => f.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match *self {
            Output::Standard(ref mut s) => s.flush(),
            Output::File(ref mut f)     => f.flush(),
        }
    }
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
    static ref PRUNE_TO: usize = parse_env("PRUNE_TO", 2_000_000);
    static ref MAX_NGRAM: usize = parse_env("MAX_NGRAM", 5);
    static ref MIN_NGRAM: usize = parse_env("MIN_NGRAM", 1);
    static ref MIN_COUNT: i64 = parse_env("MIN_COUNT", 5);
    static ref MIN_SCORE: f64 = parse_env("MIN_SCORE", 0.1f64);
    static ref MAX_EXPORT: u32 = parse_env("MAX_EXPORT", 250_000);
    static ref NGRAM_DELIM: String = std::env::var("NGRAM_DELIM").unwrap_or(" ".to_string());
    static ref BATCH_SIZE: u64 = parse_env("BATCH_SIZE", 1_000_000);

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
            _ => panic!(r#"Invalid stemming language, please let LANG one of the following:
    "ar" => Arabic
    "da" => Danish
    "nl" => Dutch
    "en" => English
    "fi" => Finnish
    "fr" => French
    "de" => German
    "el" => Greek
    "hi" => Hungarian
    "it" => Italian
    "pt" => Portuguese
    "ro" => Romanian
    "ru" => Russian
    "es" => Spanish
    "sv" => Swedish
    "ta" => Tamil
    "tk" => Turkish
"#),
    };
        Stemmer::create(algorithm)
    };
    static ref TOKEN_REGEX: Regex = Regex::new(&std::env::var("TOKEN_REGEX").unwrap_or(r"\b[\w+'’]+\b".to_string())).unwrap();
    static ref CHUNK_SPLIT_REGEX: Regex = Regex::new(&std::env::var("CHUNK_SPLIT_REGEX").unwrap_or(r"[\.\?!\(\);]+".to_string())).unwrap();
}

lazy_static! {
    static ref SCORES: HashMap<Option<String>, NGramScores> = {
        let mut label_ngrams = HashMap::new();
        read_partition_scores_for_labels(&Some(list_score_labels().unwrap()), &mut label_ngrams).expect("Unable to read partitions for labels");
        label_ngrams
    };

    static ref FS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::from(());
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

/// Hashes an ngram, does stem for you.
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
                labels.push(label.to_owned());
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

#[post("/transform?<delim>", data = "<data>")]
fn api_transform(delim: Option<String>, data: Json<ApiTransformRequest>) -> JsonValue {
    let mut transformed_docs: Vec<Document> = vec!();
    let delim = delim.unwrap_or("_".to_string());

    for doc in data.0.documents {
        let labels = doc.labels.unwrap_or(vec!(None));
        let transformed = transform_text(&delim, &labels, doc.text);
        transformed_docs.push(Document {
            text: transformed,
            labels: Some(labels),
        });
    }

    json!(transformed_docs)
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
        .mount("/", routes![api_list_labels, api_analyze, api_transform])
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

fn merge_into_label_counts(label: &Option<String>, new_ngrams: NGramCounts) -> std::io::Result<()> {
    let _lock = FS_LOCK.lock().expect("Couldn't take file system lock");
    let path = match label {
        Some(label) => format!("data/counts_label={}.csv", label),
        None => String::from("data/counts_default.csv"),
    };

    let mut ngrams = {
        if let Ok(mut file) = File::open(path.clone()) {
            read_partition_counts_from_file(&mut file)?.unwrap_or(NGramCounts::new())
        } else { NGramCounts::new() }
    };

    let mut file = File::create(path)?;
    if ngrams.len() > 0 {
        merge_ngrams_into(&new_ngrams, &mut ngrams);
        write_partition_counts(&mut file, &ngrams)?;
    } else {
        write_partition_counts(&mut file, &new_ngrams)?;
    }
    Ok(())
}

fn update_phrase_model(label: Option<String>, documents: &mut Vec<String>) -> std::io::Result<()> {
    documents.sort();
    documents.dedup();

    let new_ngrams = count_ngrams(&documents);
    merge_into_label_counts(&label, new_ngrams)
}

fn update_phrase_models_from_labeled_documents(labeled_documents: &mut Vec<Document>) -> std::io::Result<()> {
    let mut groups: HashMap<Option<String>, Vec<String>> = HashMap::new();
    for labeled_document in labeled_documents.iter() {
        for label in labeled_document.labels.clone().unwrap_or(vec!(None)).iter() {
            if let Some(group) = groups.get_mut(&label) {
                group.push(labeled_document.text.clone());
            } else {
                groups.insert(label.clone(), vec!(labeled_document.text.clone()));
            }
        }
    }

    debug!("Counting ngrams for labels: {:?}", groups.keys());

    groups.par_iter_mut().for_each(move |(label, mut documents)| {
        let label = normalize_label(label);
        update_phrase_model(label.clone(), &mut documents).expect("Thread failed in phrase model update");
    });

    Ok(())
}

fn merge_ngrams_into(from: &NGramCounts, into: &mut NGramCounts) {
    debug!("Merging {} ngrams into {} ngrams.", from.len(), into.len());
    for (ngram, count) in from {
        *into.entry(*ngram).or_insert(0) += count;
    }
}

fn merge_ngrams_into_owned(from: NGramCounts, into: &mut NGramCounts) {
    for (ngram, count) in from {
        *into.entry(ngram).or_insert(0) += count;
    }
}

fn count_ngrams_into(documents: &Vec<String>, ngrams: &mut NGramCounts) {
    let max_ngram = MAX_NGRAM.clone();
    let mut doc_count = 0;
    let doc_ngrams: Vec<Vec<NGram>> = documents.par_iter()
        .map(move |document| extract_document_ngrams(document, &max_ngram))
        .collect();
    for doc_ngrams in doc_ngrams {
        for ngram in doc_ngrams {
            *ngrams.entry(ngram).or_insert(0) += 1;
        }
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

fn extract_document_ngrams(document: &String, max_ngram: &usize) -> Vec<NGram> {
    let mut doc_ngrams: Vec<NGram> = vec!();
    let delim = NGRAM_DELIM.clone();

    for chunk in CHUNK_SPLIT_REGEX.split(&document) {
        let mut ngrams: [NGram; 10] = [NGram::new(); 10];
        let mut head = 0;
        let chunk_bytes = chunk.as_bytes();

        for token in TOKEN_REGEX.find_iter(chunk) {
            if let Some(preceding_byte) = chunk_bytes.get(token.start() - 1) {
                if preceding_byte == &64 /* @ */ || preceding_byte == &46 /* . */ || preceding_byte == &47 /* / */ || preceding_byte == &92 /* \ */ {
                    // Ignore tokens that are preceded by . or @ (since rust regex doesn't do look behind)
                    continue;
                }
            }
            if let Some(following_byte) = chunk_bytes.get(token.end() + 1) {
                if following_byte == &64 /* @ */ || following_byte == &46 /* . */ || following_byte == &47 /* / */ || following_byte == &92 /* \ */ {
                    // Ignore tokens that are followed by . or @ (since rust regex doesn't do look behind)
                    continue;
                }
            }
            let token_string: String = token.as_str().to_string();

            if let Ok(_num) = token_string.parse::<i64>() {
                // Ignore numbers
                continue;
            }

            for idx in 0..*max_ngram {
                let position = (head + idx) % 10;
                // TODO: handle "buffer too small" errors here more gracefully - realistically we don't care about phrases that are too long
                if let Err(_e) = ngrams[position].try_push_str(token.as_str()) {
                    continue;
                } else {
                    doc_ngrams.push(ngrams[position]);
                    ngrams[position].try_push_str(&delim);
                }
            }
            ngrams[(head + max_ngram) % 10] = NGram::new();
            head += 1;
        }
    }

    doc_ngrams.sort();
    doc_ngrams.dedup();
    doc_ngrams
}

fn prune_ngrams(ngrams: &mut NGramCounts) {
    let ngrams_len = ngrams.len();
    if ngrams_len > PRUNE_AT.clone() {
        debug!("Pruning ngrams of length {}.", ngrams_len);
        let prune_to = PRUNE_TO.clone();
        let mut prune_below: i64 = 1;
        while ngrams.len() > prune_to {
            debug!("Pruning ngrams with count {}", prune_below);
            ngrams.retain(|_ngram, count| count > &mut prune_below);
            prune_below += 1;
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
        read_partition_counts_from_file(&mut File::open(path)?)
    } else {
        Ok(None)
    }
}

fn read_partition_counts_from_file(file: &mut File) -> std::io::Result<Option<NGramCounts>> {
    let mut reader = csv::Reader::from_reader(file);
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

fn write_partition_counts(file: &mut File, ngrams: &NGramCounts) -> std::io::Result<()> {
    let mut writer = csv::Writer::from_writer(file);
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

fn header_indexes<R: std::io::Read>(reader: &mut csv::Reader<R>, fields: &Vec<String>) -> Vec<usize> {
    let mut indexes = vec!();
    for (idx, header) in reader.headers().expect("Couldn't read CSV reader header").iter().enumerate() {
        if fields.contains(&header.to_string()) {
            indexes.push(idx);
        }
    }
    indexes
}

fn read_documents_from_csv(csv_reader: &mut csv::Reader<Input>, text_fields: &Vec<String>, label_fields: &Vec<String>, limit: &u64) -> Result<Vec<Document>, ()> {
    let text_idxs: Vec<usize> = header_indexes(csv_reader, text_fields);
    let label_idxs: Vec<usize> = header_indexes(csv_reader, label_fields);
    if text_idxs.is_empty() {
        error!("Didn't find text_field in the CSV header!");
        return Err(());
    }
    if label_idxs.is_empty() {
        error!("Didn't find label_field in the CSV header!");
        return Err(());
    }
    let mut documents = vec!();
    for record in csv_reader.records().take(limit.clone() as usize) {
        let record = record.expect("Couldn't parse row");
        let labels: Vec<Option<String>> = label_idxs.iter().map(|idx| record.get(idx.clone()).map(|s| s.to_string())).collect();
        for text_idx in text_idxs.iter() {
            if let Some(text) = record.get(text_idx.clone()) {
                documents.push(Document {
                    text: text.to_string(), 
                    labels: Some(labels.clone()),
                });
            } else {
                error!("Couldn't get column {} from CSV", text_idx);
            }
        }
        
    }
    Ok(documents)
}

fn read_documents_from_json(reader: &mut BufReader<Input>, text_fields: &Vec<String>, label_fields: &Vec<String>, limit: &u64) -> Result<Vec<Document>, ()> {
    let mut documents: Vec<Document> = vec!();
    for line in reader.lines().take(limit.clone() as usize) {
        if let Ok(line) = line {
            if let Ok(object) = serde_json::from_str(&line[..]) {
                let object: serde_json::Value = object;
                let labels: Vec<Option<String>> = label_fields.iter().map(|l| object[l].as_str().map(|s| s.to_string())).collect();
                for text_field in text_fields {
                    let text = object[text_field].as_str();
                    documents.push(Document {
                        labels: Some(labels.clone()),
                        text: text.expect("Expected to find text").to_string(),
                    });
                }
            } else {
                error!("Unable to parse JSON object");
                return Err(());
            }
        } else {
            error!("Unable to read line from buffer");
            return Err(());
        }
    }
    Ok(documents)
}

fn read_documents_from_plain(reader: &mut BufReader<Input>, labels: Vec<Option<String>>, limit: &u64) -> std::io::Result<Vec<Document>> {
    let mut documents: Vec<Document> = vec!();
    for line in reader.lines().take(limit.clone() as usize) {
        let line: String = line?;
        documents.push(Document {
            text: line.clone(),
            labels: Some(labels.iter().map(|s| s.clone().map(|ss| ss.to_string())).collect()),
        });
    }
    Ok(documents)
}

fn count_stdin(labels: Vec<Option<String>>, mode: ParseMode, text_fields: Option<Vec<String>>, label_fields: Option<Vec<String>>) {
    let text_fields = text_fields.unwrap_or(vec!("text".to_string()));
    let label_fields = label_fields.unwrap_or(vec!("label".to_string()));
    let mut batch_reader = BatchedInputReader::new(Input::stdin(), mode, BATCH_SIZE.clone(), text_fields, label_fields, labels.clone());

    loop {
        let documents = batch_reader.read_batch();
        if let Some(mut documents) = documents {
            debug!("Process batch of {} documents.", documents.len());
            update_phrase_models_from_labeled_documents(&mut documents).expect("Failed to update phrase models.");
        } else {
            break;
        }
    }
}

fn count_file(path: &str, labels: Vec<Option<String>>, mode: ParseMode, text_fields: Option<Vec<String>>, label_fields: Option<Vec<String>>) {
    let text_fields = text_fields.unwrap_or(vec!("text".to_string()));
    let label_fields = label_fields.unwrap_or(vec!("label".to_string()));
    let batch_reader = BatchedInputReader::new(Input::file(path.to_string()).expect("Couldn't open file for counting ngrams"), mode, BATCH_SIZE.clone(), text_fields, label_fields, labels.clone());

    batch_reader.par_bridge().for_each(|mut documents| {
        debug!("Process batch of {} documents.", documents.len());
        update_phrase_models_from_labeled_documents(&mut documents).expect("Failed to update phrase models.");
    });
}

fn cmd_count(path: &str, labels: Vec<Option<String>>, mode: ParseMode, text_fields: Option<Vec<String>>, label_fields: Option<Vec<String>>) {
    if mode != ParseMode::PlainText && !labels.is_empty() && labels[0] != None {
        error!("Cannot specify labels for non-plaintext parse mode");
        std::process::exit(1);
    } else if mode == ParseMode::PlainText && text_fields.is_some() {
        error!("Can't specify text field for non-CSV input");
        std::process::exit(1);
    } else if mode == ParseMode::PlainText && label_fields.is_some() {
        error!("Can't specify label field for non-CSV input");
        std::process::exit(1);
    }
    std::fs::create_dir_all("data").expect("Failed to ensure data directory existence.");
    if path == "-" {
        count_stdin(labels, mode, text_fields, label_fields);
    } else {
        count_file(path, labels, mode, text_fields, label_fields);
    };
}

fn cmd_transform(input: String, output: Option<String>, mode: ParseMode, delim: String, labels: Vec<Option<String>>, text_fields: Option<Vec<String>>, label_fields: Option<Vec<String>>) {
    transform_inner(
        &mut Input::from_arg(Some(input)).expect("Couldn't open input"),
        &mut Output::from_arg(output).expect("Couldn't open output"),
        mode,
        delim,
        labels,
        text_fields,
        label_fields,
    );
}

fn transform_inner(inbuf: &mut Input, outbuf: &mut Output, mode: ParseMode, delim: String, labels: Vec<Option<String>>, text_fields: Option<Vec<String>>, label_fields: Option<Vec<String>>) {
    let label_fields = match label_fields { Some(fields) => fields, None => vec!() };
    let text_fields = match text_fields { Some(fields) => fields, None => vec!() };
    if mode == ParseMode::CSV {
        transform_csv(inbuf, outbuf, delim, &text_fields, &label_fields);
    } else if mode == ParseMode::PlainText {
        transform_plain(inbuf, outbuf, delim, labels);
    } else if mode == ParseMode::JSON {
        transform_json(inbuf, outbuf, delim, &text_fields, &label_fields);
    }
}

fn transform_csv(inbuf: &mut std::io::Read, outbuf: &mut std::io::Write, delim: String, text_fields: &Vec<String>, label_fields: &Vec<String>) {
    let mut reader = csv::Reader::from_reader(inbuf);
    let mut writer = csv::Writer::from_writer(outbuf);
    let text_idxs: Vec<usize> = header_indexes(&mut reader, text_fields);
    let label_idxs: Vec<usize> = header_indexes(&mut reader, label_fields);
    for record in reader.records() {
        let record = record.expect("Couldn't read row from CSV");
        let mut out_record: Vec<String> = vec!();
        let labels: Vec<Option<String>> = label_idxs.iter().map(|l| record.get(l.clone()).map(|s| s.to_string())).collect();
        for (idx, val) in record.iter().enumerate() {
            if text_idxs.contains(&idx) {
                out_record.push(transform_text(&delim, &labels, val.to_string()));
            } else {
                out_record.push(val.to_string());
            }
        }
        writer.write_record(out_record.iter()).expect("Couldn't write CSV row");
    }
}

fn transform_plain(inbuf: &mut std::io::Read, outbuf: &mut std::io::Write, delim: String, labels: Vec<Option<String>>) {
    let reader = BufReader::new(inbuf);
    for line in reader.lines() {
        writeln!(outbuf, "{}", transform_text(&delim, &labels, line.expect("Couldn't read plaintext line"))).expect("Couldn't write plaintext output");
    }
    outbuf.flush().expect("Couldn't flush output buffer");
}

fn transform_json(inbuf: &mut std::io::Read, outbuf: &mut std::io::Write, delim: String, text_fields: &Vec<String>, label_fields: &Vec<String>) {
    let reader = BufReader::new(inbuf);
    for line in reader.lines() {
        if let Ok(object) = serde_json::from_str(&line.expect("Couldn't read JSON line")[..]) {
            let mut object: serde_json::Value = object;
            let labels: Vec<Option<String>> = label_fields.iter().map(|l| object[l].as_str().map(|s| s.to_string())).collect();
            for text_field in text_fields {
                if let Some(text) = object[text_field].as_str() {
                    object[text_field] = serde_json::value::Value::String(transform_text(&delim, &labels, text.to_string()));
                }
            }
            write!(outbuf, "{}", serde_json::to_string(&object).expect("Couldn't dump object to json")).expect("Couldn't write JSON record");
        }
    }
    outbuf.flush().expect("Couldn't flush final JSON output");
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


fn transform_text(delim: &String, labels: &Vec<Option<String>>, document: String) -> String {
    labels.iter().fold(document, |document, label| transform_text_inner(delim, label, &document))
}

/// Eager implementation of phrase transform - as long as the the deque contains a phrase, keep trying to add more tokens
/// Example:
///  In:  'Please, use the fax machine to send it.'
///  Out: 'Please, use the fax_machine to send it.'
fn transform_text_inner(delim: &String, label: &Option<String>, document: &String) -> String {
    let max_ngram = MAX_NGRAM.clone();
    let min_score = MIN_SCORE.clone();
    let mut result = String::new();
    let mut current_phrase: VecDeque<Match> = VecDeque::with_capacity(max_ngram);
    let mut last_token_end = 0;
    let label = normalize_label(label);
    for token in TOKEN_REGEX.find_iter(document) {
        current_phrase.push_back(token);
        if current_phrase.len() == max_ngram {
            result.push_str(&document[last_token_end..current_phrase.get(0).expect("Should have been able to get head token").start()]);
            // At this point, the result string is up to date (apart from stems in the queue already)
            let mut stem_window: Vec<String> = current_phrase.iter().map(|m| re_match_stem(m.clone())).collect();
            let tokens_written = allocate_ngrams(&mut stem_window, &mut result, &label, &min_score, delim);
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

    // Handle the last few tokens (which may be a phrase!)
    while !current_phrase.is_empty() {
        result.push_str(&document[last_token_end..current_phrase.get(0).expect("Should have been able to get head token").start()]);
        let mut stem_window: Vec<String> = current_phrase.iter().map(|m| re_match_stem(m.clone())).collect();
        let tokens_written = allocate_ngrams(&mut stem_window, &mut result, &label, &min_score, delim);
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

fn normalize_label(label: &Option<String>) -> Option<String> {
    if let Some(label) = label {
        let label = Regex::new(r"\&").unwrap().replace_all(label, "and").to_string();
        let label = Regex::new(r"[<>\|\\:\(\)&;]").unwrap().replace_all(&label, "").to_string();
        Some(label)
    } else {
        None
    }
}

fn main() {
    let matches = clap_app!(phrase => 
        (version: "0.3.4")
        (author: "Stuart Axelbrooke <stuart@axelbrooke.com>")
        (about: "Detect phrases in free text data.")
        (setting: clap::AppSettings::ArgRequiredElseHelp)
        (@subcommand count =>
            (about: "Count ngrams in provided input text data")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value ... "Label to apply to the provided documents")
            (@arg labelfield: --labelfield +takes_value ... "The field to use for labeling text")
            (@arg textfield: --textfield +takes_value ... "The text field to use to learn phrases")
            (@arg mode: -m --mode +takes_value "Input parse mode, uses `label` and `text` field by default, or use `labelfield` or `textfield` to specify fields to take labels or texts from.")
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
            (@arg labelfield: --labelfield +takes_value ... "The field to use for labeling text")
            (@arg textfield: --textfield +takes_value ... "The text field to use to learn phrases")
            (@arg mode: -m --mode +takes_value "Input parse mode, uses `label` and `text` field by default, or use `labelfield` or `textfield` to specify fields to take labels or texts from.")
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
        let mode: Result<ParseMode, ()> = matches.value_of("mode").unwrap_or("plain").parse();
        let mode: ParseMode = mode.expect("Invalid parse mode provided.");
        let labels: Option<Vec<String>> = matches.values_of("label").map(|v| v.map(|s| s.to_string()).collect());
        let labels: Vec<Option<String>> = if let Some(labels) = labels {
            if labels.is_empty() {
                vec!(None)
            } else {
                labels.iter().map(|l| Some(l.clone())).collect()
            }
        } else {
            vec!(None)
        };
        if let Some(num_workers) = matches.value_of("workers").map(|s| s.parse::<usize>().expect("Couldn't parse --workers")) {
            std::env::set_var("RAYON_NUM_THREADS", num_workers.to_string());
        }
        let text_fields: Option<Vec<String>> = matches.values_of("textfield").map(|v| v.map(|s| s.to_string()).collect());
        let label_fields: Option<Vec<String>> = matches.values_of("labelfield").map(|v| v.map(|s| s.to_string()).collect());
        match matches.value_of("input") {
            Some(path) => {
                cmd_count(path, labels, mode, text_fields, label_fields);
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
        let delim = matches.value_of("delim").map(|s| s.to_string()).unwrap_or("_".to_string());
        let output = matches.value_of("output").map(|s| s.to_string());
        let mode: Result<ParseMode, ()> = matches.value_of("mode").unwrap_or("plain").parse();
        let mode: ParseMode = mode.expect("Invalid parse mode provided.");
        let labels: Option<Vec<String>> = matches.values_of("label").map(|v| v.map(|s| s.to_string()).collect());
        let labels: Vec<Option<String>> = if let Some(labels) = labels {
            if labels.is_empty() {
                vec!(None)
            } else {
                labels.iter().map(|l| normalize_label(&Some(l.clone()))).collect()
            }
        } else {
            vec!(None)
        };
        let labels = labels.iter().map(|label| normalize_label(&label)).collect();
        let text_fields: Option<Vec<String>> = matches.values_of("textfield").map(|v| v.map(|s| s.to_string()).collect());
        let label_fields: Option<Vec<String>> = matches.values_of("labelfield").map(|v| v.map(|s| s.to_string()).collect());
        match matches.value_of("input") {
            Some(input) => {
                cmd_transform(input.to_string(), output, mode, delim, labels, text_fields, label_fields);
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
