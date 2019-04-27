#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;
extern crate rocket_contrib;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
#[macro_use] extern crate log;
extern crate rusqlite;
extern crate env_logger;
// extern crate native_stemmers; // see https://crates.io/crates/rust-stemmers

// use rocket_contrib::json::Json;
use std::io::{BufReader, BufRead};
use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::io::prelude::*;
use std::io::LineWriter;
use std::path::Path;
use std::fs::File;

const DB_PATH: &str = "database.sqlite";
const PRUNE_AT: usize = 1_000_000;
const MAX_NGRAM: usize = 4;
const FILE_DELIM: char = '\t';
const MIN_COUNT: i64 = 5;
const MAX_EXPORT: u32 = 250_000;

// URL regex
// [-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)

// Token regex
// ([\w']+)

// Sentence split pattern
// [\.\?\!\(\)\;]

#[derive(Deserialize, Debug)]
struct Document {
    labels: Option<Vec<String>>,
    body: String,
}

#[get("/documents/statistics/<label>")]
fn get_document_statistics(label: Option<String>) -> String {
    format!("Total documents: {}", -1)
}


#[get("/hello")]
fn hello() -> String {
    String::from("Hey there!")
}

fn serve(port: u16) {
    let config = rocket::Config::build(rocket::config::Environment::Development)
        .port(port).finalize().expect("Couldn't create config.");
    rocket::custom(config)
        .mount("/", routes![hello, get_document_statistics])
        .launch();
}

fn read_partition_counts_for_labels(labels: &Option<Vec<String>>, label_ngrams: &mut HashMap<Option<String>, HashMap<Vec<String>, i64>>) -> std::io::Result<()> {
    if let Some(labels) = labels {
        for label in labels {
            let ngrams = match read_partition_counts(Some(&label)) {
                Ok(Some(ngrams)) => ngrams,
                Ok(None) => HashMap::new(),
                Err(err) => return Err(err),
            };
            label_ngrams.insert(Some(label.to_owned().to_owned()), ngrams);
        }
    } else {
        let ngrams = match read_partition_counts(None) {
            Ok(Some(ngrams)) => ngrams,
            Ok(None) => HashMap::new(),
            Err(err) => return Err(err),
        };
        label_ngrams.insert(None, ngrams);
    }
    Ok(())
}

fn update_phrase_models(labels: Option<Vec<String>>, documents: Vec<String>) -> std::io::Result<()> {
    let mut label_ngrams: HashMap<Option<String>, HashMap<Vec<String>, i64>> = HashMap::new();
    read_partition_counts_for_labels(&labels, &mut label_ngrams)?;

    if let Some(labels) = labels {
        let counted_ngrams = count_ngrams(&documents);
        for label in labels {
            merge_ngrams_into(&counted_ngrams, label_ngrams.get_mut(&Some(label)).unwrap());
        }
    } else {
        merge_ngrams_into(&count_ngrams(&documents), label_ngrams.get_mut(&None).unwrap());
    }

    for (label, ngrams) in label_ngrams {
        write_partition_counts(label.as_ref(), &ngrams)?;
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

fn count_ngrams(documents: &Vec<String>) -> HashMap<Vec<String>, i64> {
    let mut ngrams = HashMap::new();
    info!("Counting ngrams for {} documents.", documents.len());

    for document in documents {
        count_document_ngrams(&document, &mut ngrams);
    }

    ngrams
}

fn count_document_ngrams(document: &String, ngrams: &mut HashMap<Vec<String>, i64>) {
    let re = Regex::new("[\\w']+").unwrap();
    let mut token_queues: Vec<VecDeque<String>> = Vec::new();
    for i in 1..MAX_NGRAM {
        token_queues.push(VecDeque::with_capacity(i));
    }

    for token in re.find_iter(document) {
        let token_string = token.as_str().to_string();
        for i in 1..MAX_NGRAM {
            if let Some(queue) = token_queues.get_mut(i - 1) {
                queue.push_back(token_string.to_owned());
                if queue.len() > i {
                    queue.pop_front();
                }
                if queue.len() == i {
                    let queue: Vec<String> = queue.iter().map(|s| s.to_string()).collect();
                    if let Some(count) = ngrams.get_mut(&queue) {
                        *count += 1;
                    } else {
                        ngrams.insert(queue, 1);
                    }
                }
            }
        }
    }
}

fn read_partition_counts(label: Option<&String>) -> std::io::Result<Option<HashMap<Vec<String>, i64>>> {
    let path_str: String = match label {
        Some(label) => format!("data/counts_label={}", label),
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

fn write_partition_counts(label: Option<&String>, ngrams: &HashMap<Vec<String>, i64>) -> std::io::Result<()> {
    info!("Writing partition {:?}.", &label);
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

fn analyze_stdin(labels: Option<Vec<String>>) {
    let stdin = std::io::stdin();
    let documents: Vec<String> = stdin.lock().lines().map(|s| s.expect("Coulnd't read line")).collect();
    update_phrase_models(labels, documents).expect("Failed to update phrase models");
}

fn analyze_file(path: &str, labels: Option<Vec<String>>) {
    let file = std::fs::File::open(path).unwrap();
    let documents: Vec<String> = BufReader::new(file).lines().map(|s| s.expect("Couldn't read line")).collect();
    update_phrase_models(labels, documents).expect("Failed to update phrase models");
}

fn cmd_analyze(path: &str, labels: Option<Vec<String>>) {
    if path == "-" {
        analyze_stdin(labels);
    } else {
        analyze_file(path, labels);
    };
}

fn cmd_export(path: Option<&str>, label: Option<String>) {
    info!("Exporting for label {:?}", label);
    let ngrams = read_partition_counts(label.as_ref())
        .expect("Couldn't read phrase counts.")
        .expect("No phrase counts found for that label.");
    
    let mut scores: HashMap<Vec<String>, f64> = HashMap::new();
    let total_count: i64 = ngrams.values().map(|i| i.to_owned()).sum();
    let total_count: f64 = total_count as f64;
    for (ngram, ngram_count) in &ngrams {
        if ngram.len() == 1{
            continue;
        }

        // // Skipgram Strategy
        // let vocab_size = ngrams.len() as f64;
        // let float_count = *ngram_count as f64;
        // let mut score: f64 = (ngram_count - MIN_COUNT) as f64 * vocab_size;

        // for unigram in ngram {
        //     if let Some(unigram_count) = ngrams.get(&vec!(unigram.to_string())) {
        //         score /= unigram_count.clone() as f64;
        //     }
        // }

        // score = score.ln() / -(float_count / vocab_size).ln();

        // NPMI Strategy
        if ngram_count > &0i64 {
            let pj: f64 = ngram_count.clone() as f64 / total_count;
            let mut score: f64 = pj;
            
            for unigram in ngram {
                if let Some(unigram_count) = ngrams.get(&vec!(unigram.to_string())) {
                    score /= unigram_count.clone() as f64;
                }
            }
            score = score.ln();

            score /= -pj.ln();

            scores.insert(ngram.clone(), score);
        }
    }

    let mut scores: Vec<(&Vec<String>, &f64)> = scores.iter().filter(|(_ngram, score)| {
        score.is_finite() && score > &&0.15f64
    }).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let mut written = 0;

    if let Some(path) = path {
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
    } else {
        for (ngram, score) in scores {
            let phrase = ngram.iter().map(|s| s.to_string()).collect::<Vec<String>>().join(" "); 
            println!("{}{}{}", phrase, FILE_DELIM, score);
            
            written += 1;
            if written > MAX_EXPORT {
                return;
            }
        }
    }
}

fn main() {
    let matches = clap_app!(app => 
        (version: "0.1")
        (author: "Stuart Axelbrooke <stuart@axelbrooke.com>")
        (about: "Detect phrases in free text data.")
        (@subcommand analyze =>
            (about: "Analyze provided input text data")
            (@arg input: +required "File to read text data from, use - to pipe from stdin")
            (@arg label: -l --label +takes_value ... "Labels to apply to the documents")
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
    } else if let Some(matches) = matches.subcommand_matches("analyze") {
        env_logger::init();
        let is_csv = matches.value_of("csv").is_some();
        let labels = matches.values_of("label").map(|values| values.map(|s| s.to_string()).collect());
        match matches.value_of("input") {
            Some(path) => {
                cmd_analyze(path, labels);
            },
            None => {
                error!("Must provide a file to read text from, or pass - and stream to stdin.");
                std::process::exit(1);
            }
        }
    } else if let Some(matches) = matches.subcommand_matches("export") {
        env_logger::init();
        let label = matches.value_of("label");
        let output = matches.value_of("output");
        cmd_export(output, label.map(|s| s.to_string()));
    }
}
