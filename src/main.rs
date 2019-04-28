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

// use rocket_contrib::json::Json;
use std::io::{BufReader, BufRead};
use regex::Regex;
use std::collections::{HashMap, VecDeque, HashSet};
use std::io::prelude::*;
use std::io::LineWriter;
use std::path::Path;
use std::fs::File;

use rocket_contrib::json::{Json, JsonValue};

const DB_PATH: &str = "database.sqlite";
const PRUNE_AT: usize = 1_000_000;
const MAX_NGRAM: usize = 4;
const FILE_DELIM: char = '\t';
const MIN_COUNT: i64 = 5;
const MAX_EXPORT: u32 = 250_000;
const MIN_SCORE: f64 = 0.1;

// URL regex
// [-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)

// Token regex
// ([\w']+)

// Sentence split pattern
// [\.\?\!\(\)\;]

#[derive(Deserialize, Debug)]
struct Document {
    label: Option<String>,
    body: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AnalyzedDocument {
    label: Option<String>,
    body: String,
    ngrams: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct ApiAnalyzeRequest {
    documents: Vec<Document>
}

lazy_static! {
    static ref HEAD_UNIGRAM_IGNORES: HashSet<String> = vec!(
        "the", "a", "is", "and", "of", "to",
    ).iter().map(|s| s.to_string()).collect();

    static ref TAIL_UNIGRAM_IGNORES: HashSet<String> = vec!(
        "the", "a", "i", "is", "you", "and", "my", "so",
    ).iter().map(|s| s.to_string()).collect();

    static ref TOKEN_REGEX: Regex = Regex::new("[\\w+'’]+").unwrap();
}

lazy_static! {
    static ref SCORES: HashMap<Option<String>, HashMap<Vec<String>, f64>> = {
        let mut label_ngrams = HashMap::new();
        read_partition_scores_for_labels(&Some(list_score_labels().unwrap()), &mut label_ngrams).expect("Unable to read partitions for labels");
        label_ngrams
    };
}

fn analyze_text(text: &String, scores: &HashMap<Vec<String>, f64>) -> Vec<String> {
    let mut significant_ngrams: Vec<String> = vec!();
    let mut token_queues: Vec<VecDeque<String>> = Vec::new();
    for i in 1..MAX_NGRAM+1 {
        token_queues.push(VecDeque::with_capacity(i));
    }

    for token in TOKEN_REGEX.find_iter(&text.to_lowercase()) {
        let token_string = token.as_str().to_string();
        for i in 1..MAX_NGRAM+1 {
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

    significant_ngrams.sort();
    significant_ngrams.dedup();

    significant_ngrams
}

#[post("/analyze", data = "<data>")]
fn api_analyze(data: Json<ApiAnalyzeRequest>) -> JsonValue {
    let analyzed_docs: Vec<AnalyzedDocument> = data.0.documents.iter().map(|d| {
        let significant_terms: Vec<String> = if let Some(scores) = SCORES.get(&d.label) {
            analyze_text(&d.body, scores)
        } else {
            vec!()
        };
        AnalyzedDocument {
            label: d.label.to_owned(),
            body: d.body.to_owned(),
            ngrams: significant_terms,
        }
    }).collect();
    json!(analyzed_docs)
}

#[get("/hello")]
fn hello() -> String {
    String::from("Hey there!")
}

fn serve(port: u16) {
    let config = rocket::Config::build(rocket::config::Environment::Development)
        .port(port).finalize().expect("Couldn't create config.");
    rocket::custom(config)
        .mount("/", routes![hello, api_analyze])
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

    info!("Labels: {:?}", groups.keys());

    for (label, documents) in groups.iter_mut() {
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
    for document in documents {
        count_document_ngrams(&document, ngrams);
    }
}

fn count_ngrams(documents: &Vec<String>) -> HashMap<Vec<String>, i64> {
    let mut ngrams = HashMap::new();
    debug!("Counting ngrams for {} documents.", documents.len());
    count_ngrams_into(documents, &mut ngrams);
    ngrams
}

fn count_document_ngrams(document: &String, ngrams: &mut HashMap<Vec<String>, i64>) {
    let mut token_queues: Vec<VecDeque<String>> = Vec::new();
    for i in 1..MAX_NGRAM+1 {
        token_queues.push(VecDeque::with_capacity(i));
    }

    let mut unique_ngrams: HashSet<Vec<String>> = HashSet::new();

    for token in TOKEN_REGEX.find_iter(&document.to_lowercase()) {
        let token_string = token.as_str().to_string();
        for i in 1..MAX_NGRAM+1 {
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

    for ngram in unique_ngrams.iter() {
        let ngram = ngram.to_owned();
        if let Some(count) = ngrams.get_mut(&ngram) {
            *count += 1;
        } else {
            ngrams.insert(ngram, 1);
        }
    }
}

fn read_partition_counts(label: &Option<&String>) -> std::io::Result<Option<HashMap<Vec<String>, i64>>> {
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
    for (ngram, count) in counts.iter().take(PRUNE_AT) {
        let phrase = ngram.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(" "); 
        writeln!(file, "{}{}{}", phrase, FILE_DELIM, count)?;
    }
    file.flush()?;
    Ok(())
}

fn count_stdin(label: Option<String>, is_csv: bool) {
    // TODO accept stdin CSV
    let stdin = std::io::stdin();
    let mut documents: Vec<String> = stdin.lock().lines().map(|s| s.expect("Coulnd't read line")).collect();
    update_phrase_models(label, &mut documents).expect("Failed to update phrase models");
}

#[derive(Deserialize)]
struct LabeledDocument {
    label: Option<String>,
    text: String,
}

fn count_file(path: &str, label: Option<String>, is_csv: bool) {
    info!("Is CSV? {}", is_csv);
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
    if path == "-" {
        count_stdin(label, is_csv);
    } else {
        count_file(path, label, is_csv);
    };
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

            if let Some(score) = npmi_phrase_score(&total_count, ngram, ngram_count, &ngrams) {
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
                if let Some(score) = npmi_label_score(&all_tokens_total_count, &this_label_total_count, &ngram_count_across_labels, ngram, ngram_count) {
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
    let mut scores: Vec<(&Vec<String>, &f64)> = scores.iter().filter(|(_ngram, score)| {
        score.is_finite() // && score > &&0.15f64
    }).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let mut written = 0;

    let path = match label {
        Some(label) => format!("data/scores_label={}.txt", label),
        None => "data/scores_root.txt".to_string()
    };

    let file = File::create(path).unwrap();
    let mut file = LineWriter::new(file);
    for (ngram, score) in scores {
        let phrase = ngram.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(" "); 
        writeln!(file, "{}{}{}", phrase, FILE_DELIM, score).unwrap();

        written += 1;
        if written > MAX_EXPORT {
            return;
        }
    }
    file.flush().unwrap();
}

/// Calculates normalized mutual information between ngram and label
/// p_joint = ngram_count / this_label_total_count
/// pt = this_token_total_count / all_tokens_total_count
/// pl = this_label_total_count / all_labels_total_count
fn npmi_label_score(all_tokens_total_count: &f64, this_label_total_count: &f64, ngram_count_across_labels: &i64, ngram: &Vec<String>, ngram_count: &i64) -> Option<f64> {
    let pj = ngram_count.clone() as f64 / all_tokens_total_count; // this_label_total_count;
    let pt = ngram_count_across_labels.clone() as f64 / all_tokens_total_count;
    let pl = this_label_total_count / all_tokens_total_count;

    if pt > 0f64 && pl > 0f64 && ngram_count >= &MIN_COUNT {
        let score = (pj / pt / pl).ln() / -pj.ln();
        if score > 0f64 { Some(score) } else { None }
    } else {
        None
    }
}

fn npmi_phrase_score(corpus_size: &f64, ngram: &Vec<String>, ngram_count: &i64, ngrams: &HashMap<Vec<String>, i64>) -> Option<f64> {
    if ngram_count > &MIN_COUNT {
        let left_subgram: Vec<String> = ngram[..ngram.len() - 1].to_owned();//ngram.iter().take(ngrams.len() - 1).map(|s| s.to_string()).collect();
        let left_unigram: Vec<String> = ngram[..1].to_owned(); //ngram.iter().take(1).map(|s| s.to_string()).collect();
        let right_subgram: Vec<String> = ngram[1..].to_owned(); //ngram.iter().skip(1).map(|s| s.to_string()).collect();
        let right_unigram: Vec<String> = ngram[ngram.len() - 1 ..].to_owned(); //ngram.iter().skip(ngrams.len() - 1).map(|s| s.to_string()).collect();}

        let pj: f64 = ngram_count.clone() as f64 / corpus_size;
        let pau: f64 = ngrams.get(&left_unigram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pas: f64 = ngrams.get(&right_subgram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pa = (pas * pau).sqrt();
        let pbs: f64 = ngrams.get(&left_subgram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pbu: f64 = ngrams.get(&right_unigram).unwrap_or(ngram_count).clone() as f64 / corpus_size;
        let pb = (pbs * pbu).sqrt();
        let score: f64 = (pj / pa / pb).ln() / -pj.ln();

        if score > 0f64 { Some(score) } else { None }
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
    let matches = clap_app!(app => 
        (version: "0.1")
        (author: "Stuart Axelbrooke <stuart@axelbrooke.com>")
        (about: "Detect phrases in free text data.")
        (@subcommand count =>
            (about: "Count ngrams in provided input text data")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value "Labels to apply to the documents")
            (@arg csv: --csv "Parse input as CSV.  Use `label` column for label, `text` column to learn phrases.")
        )
        (@subcommand serve =>
            (about: "Start API server")
            (@arg port: -p --port +takes_value "Port to serve on (default 6220)")
        )
        (@subcommand export =>
            (about: "Export a model from the ngram counts for a given label")
            (@arg label: "The label for which to export a phrase model.")
            (@arg output: -o --output +takes_value "Where to write the phrase model.")
        )
    ).get_matches();

    if let Some(matches) = matches.subcommand_matches("serve") {
        let port = matches.value_of("port").unwrap_or("6220").parse::<u16>().expect("Couldn't parse port.");
        serve(port);
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
    }
}
