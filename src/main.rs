#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;
#[macro_use] extern crate rocket_contrib;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
#[macro_use] extern crate log;
#[macro_use] extern crate lazy_static;

extern crate rusqlite;
extern crate env_logger;
extern crate rust_stemmers; // see https://crates.io/crates/rust-stemmers

use std::io::{BufReader, BufRead};
use regex::{Regex, Match};
use std::collections::{HashMap, VecDeque, HashSet};
use std::path::Path;
use std::fs::File;
use rust_stemmers::{Algorithm, Stemmer};
use rocket_contrib::json::{Json, JsonValue};

// URL regex
// [-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)

type NGram = Vec<String>;
type StemmedNGram = NGram;

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
type NGramCounts = HashMap<StemmedNGram, Vec<NGramCount>>;
type CanonicalizedNGramCounts = HashMap<StemmedNGram, NGramCount>;
// Maps from stemmed ngram to cannonical ngram and its score
type NGramScores = HashMap<StemmedNGram, ScoredNGram>;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NGramCountRow {
    stemmed: String,
    ngram: String,
    count: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NGramScoreRow {
    stemmed: String,
    ngram: String,
    score: f64,
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


lazy_static! {
    // TODO parse from env vars
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


fn ngram_from_str(s: &String) -> NGram {
    let delim: &String = &NGRAM_DELIM;
    s.split(delim).map(|s| s.to_string()).collect()
}

fn ngram_to_str(ngram: &NGram) -> String {
    let delim: &String = &NGRAM_DELIM;
    ngram.join(delim)
}

fn analyze_text(text: &String, scores: &NGramScores, max_ngram: &usize, min_score: &f64) -> Vec<String> {
    let mut significant_ngrams: Vec<String> = vec!();

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
                        let queue: NGram = queue.iter().map(|s| s.to_string()).collect();
                        if let Some(scored_ngram) = scores.get(&queue) {
                            if &scored_ngram.score > min_score {
                                if let Some(canon_ngram) = get_canonical_ngram(&scored_ngram.ngram) {
                                    significant_ngrams.push(canon_ngram.join(" "));
                                }
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
    let min_score = MIN_SCORE.clone();
    let analyzed_docs: Vec<AnalyzedDocument> = data.0.documents.iter().map(|d| {
        let significant_terms: Vec<String> = if let Some(scores) = SCORES.get(&d.label) {
            analyze_text(&d.text, scores, &max_ngram, &min_score)
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

fn re_match_stem(re_match: Match) -> String {
    STEMMER.stem(&re_match.as_str().replace('’', "'").to_lowercase()).to_string()
}

fn get_canonical_ngram(ngram: &StemmedNGram) -> Option<NGram> {
    SCORES.get(&None).expect("No root score partition found").get(ngram).map(|ngc| ngc.ngram.clone())
}

fn read_partition_counts_for_labels(labels: &Vec<String>, label_ngrams: &mut HashMap<Option<String>, NGramCounts>) -> std::io::Result<()> {
    for label in labels {
        let ngrams = match read_partition_counts(&label) {
            Ok(Some(ngrams)) => ngrams,
            Ok(None) => HashMap::new(),
            Err(err) => return Err(err),
        };
        label_ngrams.insert(Some(label.to_owned()), ngrams);
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

fn update_phrase_models(label: String, documents: &mut Vec<String>) -> std::io::Result<()> {
    documents.sort();
    documents.dedup();
    let mut label_ngrams: HashMap<Option<String>, NGramCounts> = HashMap::new();
    read_partition_counts_for_labels(&vec!(label.clone()), &mut label_ngrams)?;
    let mut ngrams = label_ngrams.get_mut(&Some(label.clone())).unwrap();
    merge_ngrams_into(&count_ngrams(&documents), &mut ngrams);
    write_partition_counts(&label, &ngrams)?;
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
        if let Some(label) = label {
            debug!("Counting ngrams for label: {:?}", label);
            update_phrase_models(label.clone(), documents)?;
        } else {
            warn!("Ignoring unlabeled document.");
        }
    }

    Ok(())
}

fn merge_ngrams_into(from: &NGramCounts, into: &mut NGramCounts) {
    debug!("Merging {} ngrams into {} ngrams.", from.len(), into.len());
    for (stem, ngram_counts) in from {
        if let Some(into_ngram_counts) = into.get_mut(stem) {
            for ngram_count in ngram_counts.iter() {
                if let Some(mut into_ngram_count) = into_ngram_counts.iter_mut().filter(|ngc| ngc.ngram == ngram_count.ngram).next() {
                    into_ngram_count.count += ngram_count.count;
                } else {
                    into_ngram_counts.push(ngram_count.clone());
                }
            }
        } else {
            into.insert(stem.clone(), ngram_counts.clone());
        }
    }
}

fn merge_ngrams_into_owned(from: NGramCounts, into: &mut NGramCounts) {
    for (stem, ngram_counts) in from {
        if let Some(into_ngram_counts) = into.get_mut(&stem) {
            for ngram_count in ngram_counts.iter() {
                if let Some(mut into_ngram_count) = into_ngram_counts.iter_mut().filter(|ngc| ngc.ngram == ngram_count.ngram).next() {
                    into_ngram_count.count += ngram_count.count;
                } else {
                    into_ngram_counts.push(ngram_count.clone());
                }
            }
        } else {
            into.insert(stem, ngram_counts);
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
    // let mut unique_ngrams: HashSet<Vec<String>> = HashSet::new();
    let mut doc_ngrams = NGramCounts::new();

    for chunk in CHUNK_SPLIT_REGEX.split(&document) {
        let mut token_queues: Vec<VecDeque<String>> = Vec::new();
        for i in 1..max_ngram+1 {
            token_queues.push(VecDeque::with_capacity(i));
        }

        let mut stem_queues: Vec<VecDeque<String>> = Vec::new();
        for i in 1..max_ngram+1 {
            stem_queues.push(VecDeque::with_capacity(i));
        }

        for token in TOKEN_REGEX.find_iter(chunk) {
            let token_string: String = token.as_str().to_string();
            let token_stem: String = re_match_stem(token);

            for i in 1..max_ngram+1 {
                if let Some(token_queue) = token_queues.get_mut(i - 1) {
                    let stem_queue = stem_queues.get_mut(i - 1).expect("Stem queue was shorted than token queue");
                    
                    token_queue.push_back(token_string.clone());
                    if token_queue.len() > i {
                        token_queue.pop_front();
                    }
                    
                    stem_queue.push_back(token_stem.clone());
                    if stem_queue.len() > i {
                        stem_queue.pop_front();
                    }

                    if token_queue.len() == i {
                        let token_queue: Vec<String> = token_queue.iter().map(|s| s.to_string()).collect();
                        let stem_queue: Vec<String> = stem_queue.iter().map(|s| s.to_string()).collect();
                        doc_ngrams.insert(stem_queue, vec!(NGramCount { ngram: token_queue, count: 1 }));
                    }
                }
            }
        }
    }
    
    merge_ngrams_into_owned(doc_ngrams, ngrams);
}

fn prune_ngrams(ngrams: &mut NGramCounts) {
    let ngrams_len = ngrams.len();
    let num_to_prune = ngrams_len - PRUNE_TO.clone();
    if ngrams_len > PRUNE_AT.clone() {
        debug!("Pruning ngrams of length {}.", ngrams_len);

        let mut ngram_pairs: Vec<(&StemmedNGram, &Vec<NGramCount>)> = ngrams.iter().collect();
        ngram_pairs.sort_by(|a, b| {
            let a_count: i64 = a.1.iter().map(|ngc| ngc.count).sum();
            let b_count: i64 = b.1.iter().map(|ngc| ngc.count).sum();
            b_count.cmp(&a_count)
        });

        let stems_to_prune: Vec<StemmedNGram> = ngram_pairs.iter().skip(num_to_prune).map(|(stem, _ngc)| stem.to_owned().to_owned()).collect();
        for stem in stems_to_prune.iter() {
            ngrams.remove(stem);
        }
        debug!("Done pruning.");
    }
}

fn read_partition_counts(label: &String) -> std::io::Result<Option<NGramCounts>> {
    debug!("Reading counts for label={}", label);
    let path = format!("data/counts_label={}.csv", label);
    let path = Path::new(&path);
    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        let mut ngrams = NGramCounts::new();
        for row in reader.deserialize() {
            if let Ok(ngram_count) = row {
                let ngram_count: NGramCountRow = ngram_count;
                let stemmed = ngram_from_str(&ngram_count.stemmed);
                if let Some(ngc) = ngrams.get_mut(&stemmed) {
                    ngc.push(NGramCount { ngram: ngram_from_str(&ngram_count.ngram), count: ngram_count.count });
                } else {
                    ngrams.insert(stemmed, vec!(NGramCount { 
                        ngram: ngram_from_str(&ngram_count.ngram), 
                        count: ngram_count.count 
                    }));
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
        None => String::from("data/scores_root.csv"),
    };
    let path = Path::new(&path_str);
    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        let mut scores = NGramScores::new();
        for row in reader.deserialize() {
            if let Ok(row) = row {
                let ngram_score: NGramScoreRow = row;
                scores.insert(ngram_from_str(&ngram_score.stemmed), ScoredNGram { ngram: ngram_from_str(&ngram_score.ngram), score: ngram_score.score });
            }
        }

        Ok(Some(scores))
    } else {
        Ok(None)
    }
}

fn write_partition_counts(label: &String, ngrams: &NGramCounts) -> std::io::Result<()> {
    debug!("Writing partition {:?}.", &label);
    let mut writer = csv::Writer::from_path(format!("data/counts_label={}.csv", label))?;

    let mut rows: Vec<NGramCountRow> = vec!();

    for (stem, ngram_counts) in ngrams {
        for ngram_count in ngram_counts {
            rows.push(NGramCountRow { stemmed: ngram_to_str(stem), ngram: ngram_to_str(&ngram_count.ngram), count: ngram_count.count});
        }
    }

    rows.sort_by_key(|row| -row.count);

    for row in rows.iter().take(PRUNE_TO.clone()) {
        writer.serialize(row)?;
    }

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
        if let Some(label) = label {
            let stdin = std::io::stdin();
            let mut documents: Vec<String> = stdin.lock().lines().map(|s| s.expect("Coulnd't read line")).collect();
            update_phrase_models(label, &mut documents).expect("Failed to update phrase models");
        } else {
            error!("Must specify label to count ngrams into if not providing labeled documents.");
            std::process::exit(1);
        }
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
        if let Some(label) = label {
            let file = std::fs::File::open(path).unwrap();
            let mut documents: Vec<String> = BufReader::new(file).lines().map(|s| s.expect("Couldn't read line")).collect();
            update_phrase_models(label, &mut documents).expect("Failed to update phrase models");
        } else {
            error!("Must specify label to count ngrams into if not providing labeled documents.");
            std::process::exit(1);
        }
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


// Allocates the given window to known phrases as best possible.  Returns number of stems consumed.
fn allocate_ngrams(stem_window: &mut Vec<String>, buf: &mut String, label: &Option<String>, min_score: &f64, delim: &String) -> usize {
    let window_len = stem_window.len();
    if window_len > 1 {
        if is_significant(&stem_window, label, min_score) {
            if let Some(canonical) = get_canonical_ngram(&stem_window) {
                buf.push_str(&canonical.join(delim));
            } else {
                warn!("Didn't find a canonical ngram for {:?}, state str is `{}`", stem_window, buf);
                buf.push_str(&stem_window.join(delim));
            }
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
            let mut stem_window: NGram = current_phrase.iter().map(|m| re_match_stem(m.clone())).collect();
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

fn is_significant(ngram: &NGram, label: &Option<String>, min_score: &f64) -> bool {
    if let Some(scores) = SCORES.get(label) {
        if let Some(ngram_score) = scores.get(ngram) {
            return &ngram_score.score > min_score;
        }
    }
    false
}

fn list_phrase_labels() -> std::io::Result<Vec<String>> {
    let label_regex = Regex::new(r"counts_label=(.*).csv").unwrap();
    let mut labels = vec!();
    for fpath in std::fs::read_dir("data/")?.filter_map(|fpath| fpath.ok()) {
        if let Some(fname) = fpath.path().file_name() {
            let fname = fname.to_string_lossy().into_owned();
            if let Some(re_match) = label_regex.captures_iter(&fname).next() {
                labels.push(re_match[1].to_string());
            }
        }
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
    if std::path::Path::new("data/scores_root.csv").exists() {
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

    let root_ngrams = label_ngrams.remove(&None).expect("Didn't find root ngrams in loaded ngrams.");
    let labels: Vec<String> = label_ngrams.keys().filter_map(|l| l.clone()).collect();

    for label in labels.iter() {
        if let Some(ngrams) = label_ngrams.remove(&Some(label.clone())) {
            let scores = score_ngrams(ngrams, &root_ngrams);
            write_scores(&Some(label.clone()), &scores);
        }
    }

    write_scores(&None, &score_root_ngrams(root_ngrams));
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

fn load_canonicalized_ngrams(labels: &Vec<String>) -> std::io::Result<HashMap<Option<String>, CanonicalizedNGramCounts>> {
    let mut label_ngrams = HashMap::new();
    read_partition_counts_for_labels(labels, &mut label_ngrams).expect("Couldn't read partition counts");
    Ok(canonicalize_ngrams(label_ngrams))
}

fn canonicalize_ngrams(label_ngrams: HashMap<Option<String>, NGramCounts>) -> HashMap<Option<String>, CanonicalizedNGramCounts> {
    // Add root partition as the sum over labels per ngram count
    let mut root_ngrams = NGramCounts::new();

    for (_label, ngrams) in &label_ngrams {
        merge_ngrams_into(ngrams, &mut root_ngrams);
    }

    let mut canon_ngrams: CanonicalizedNGramCounts = CanonicalizedNGramCounts::new();
    for (stem, ngram_counts) in &root_ngrams {
        canon_ngrams.insert(stem.clone(), ngram_counts.iter().max_by_key(|ngc| ngc.count).expect("ngram counts shouldn't be empty").clone());
    }

    let mut canonicalized = HashMap::new();
    for (label, ngrams) in label_ngrams {
        let mut canonized_ngrams = CanonicalizedNGramCounts::new();
        for (stem, ngram_counts) in ngrams {
            let canon_ngram = canon_ngrams.get(&stem).expect("Didn't find stem in canon ngrams").ngram.clone();
            let count = ngram_counts.iter().map(|ngc| ngc.count).sum();
            canonized_ngrams.insert(stem, NGramCount { ngram: canon_ngram, count: count });
        }
        canonicalized.insert(label, canonized_ngrams);
    }

    canonicalized.insert(None, canon_ngrams);

    canonicalized
}

fn score_ngrams(ngram_counts: CanonicalizedNGramCounts, root_ngram_counts: &CanonicalizedNGramCounts) -> NGramScores {
    // Score tokens as a function of ngram_counts and root_ngram_counts
    // Score phrases as a funciton of ngram_counts alone

    let mut scores = NGramScores::new();

    let min_score = MIN_SCORE.clone();
    let min_count = MIN_COUNT.clone();

    let all_tokens_total_count: i64 = root_ngram_counts.iter()
                .filter(|(stem, _ngram_count)| stem.len() == 1)
                .map(|(_stem, ngram_count)| ngram_count.count).sum();
    let all_tokens_total_count: f64 = all_tokens_total_count as f64;

    let this_label_total_count: i64 = ngram_counts.iter()
                .filter(|(stem, _ngram_count)| stem.len() == 1)
                .map(|(_stem, ngram_count)| ngram_count.count).sum();;
    let this_label_total_count: f64 = this_label_total_count as f64;

    for (stem, ngram_count) in &ngram_counts {
        if !ngram_valid(stem) {
            continue;
        }

        // NGram intersect Label NPMI
        let ngram_count_across_labels = root_ngram_counts.get(&stem.clone()).map(|ngc| ngc.count).unwrap_or(ngram_count.count.clone());
        if let Some(mut score) = npmi_label_score(&all_tokens_total_count, &this_label_total_count, &ngram_count_across_labels, &ngram_count.count) {

            // NGram intersect Tokens NPMI
            if stem.len() > 1 {
                let phrase_score = npmi_phrase_score(&this_label_total_count, &ngram_count, &stem, &ngram_counts, &min_count);
                if let Some(phrase_score) = phrase_score {
                    // Geometric Mean - NOTE: this means if either score is less than zero, the ngram is essentially ignored
                    // score = (score * phrase_score).sqrt();
                    // Arithmetic Mean
                    score = (score + phrase_score) / 2f64;
                }
            }
            
            if score > min_score {
                scores.insert(stem.clone(), ScoredNGram { ngram: ngram_count.ngram.clone(), score: score });
            }
        }
    }

    scores
}

fn score_root_ngrams(root_ngram_counts: CanonicalizedNGramCounts) -> NGramScores {
    let mut scores = NGramScores::new();

    let min_score = MIN_SCORE.clone();
    let min_count = MIN_COUNT.clone();

    let all_tokens_total_count: i64 = root_ngram_counts.iter()
                .filter(|(stem, _ngram_count)| stem.len() == 1)
                .map(|(_stem, ngram_count)| ngram_count.count).sum();
    let all_tokens_total_count: f64 = all_tokens_total_count as f64;

    for (stem, ngram_count) in &root_ngram_counts {
        if !ngram_valid(stem) {
            continue;
        }
        
        if stem.len() > 1 {
            let phrase_score = npmi_phrase_score(&all_tokens_total_count, &ngram_count, &stem, &root_ngram_counts, &min_count);
            if let Some(phrase_score) = phrase_score {
                if phrase_score > min_score {
                    scores.insert(stem.clone(), ScoredNGram { ngram: ngram_count.ngram.clone(), score: phrase_score });
                }
            }
        }
    }

    scores
}

fn write_scores(label: &Option<String>, scores: &NGramScores) {
    let max_export = MAX_EXPORT.clone();
    let mut scores: Vec<NGramScoreRow> = scores.iter().filter_map(|(ngram, ngram_score)| {
        if ngram_score.score.is_finite() {
            Some(NGramScoreRow { ngram: ngram_to_str(&ngram_score.ngram), stemmed: ngram_to_str(&ngram), score: ngram_score.score })
        } else {
            None
        }
    }).collect();
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut written = 0;

    let path = match label {
        Some(label) => format!("data/scores_label={}.csv", label),
        None => "data/scores_root.csv".to_string(),
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

fn npmi_phrase_score(corpus_size: &f64, ngram_count: &NGramCount, stem: &StemmedNGram, ngrams: &CanonicalizedNGramCounts, min_count: &i64) -> Option<f64> {
    let count = ngram_count.count;
    if &count > min_count {
        let left_subgram: Vec<String> = stem[..stem.len() - 1].to_owned();
        let left_unigram: Vec<String> = stem[..1].to_owned();
        let right_subgram: Vec<String> = stem[1..].to_owned();
        let right_unigram: Vec<String> = stem[stem.len() - 1 ..].to_owned();

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

fn show_label(label: &String, num: &usize) {
    println!("\nLabel={}", label);
    let file = File::open(format!("data/scores_label={}.csv", label)).expect("Couldn't read scores file.");
    let reader = BufReader::new(file);
    for line in reader.lines().take(*num + 1) {
        println!("{}", line.expect("Couldn't read line from scores file"));
    }
}

fn cmd_show(labels: Option<Vec<String>>, num: usize) {
    let labels = labels
        .unwrap_or(list_score_labels().expect("Couldn't read scored labels").iter().filter_map(|s| s.to_owned()).collect());

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
    } else if let Some(matches) = matches.subcommand_matches("show") {
        let labels: Option<Vec<String>> = matches.values_of("label").map(|v| v.map(|s| s.to_string()).collect());
        let num: usize = matches.value_of("num").map(|s| s.parse::<usize>().expect("Couldn't parse --num")).unwrap_or(5);
        cmd_show(labels, num);
    }
}
