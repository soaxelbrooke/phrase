#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;
extern crate rocket_contrib;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
#[macro_use] extern crate log;
extern crate rusqlite;
extern crate env_logger;
// extern crate native_stemmers; // see https://crates.io/crates/rust-stemmers

use rocket_contrib::json::Json;
use rusqlite::{Connection, params, NO_PARAMS};
use std::io::{BufReader, BufRead};
use regex::Regex;
use std::collections::{HashMap, VecDeque};

const DB_PATH: &str = "database.sqlite";
const PRUNE_AT: u32 = 1_000_000;
const MAX_NGRAM: usize = 4;

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

fn setup_database() -> rusqlite::Result<Connection> {
    let conn = Connection::open(DB_PATH)?;

    conn.pragma_update(None, "journal_mode", &"wal")?;

    conn.execute(r#"
        create table if not exists documents (
            label text,
            body text not null
        );
    "#, NO_PARAMS)?;

    // TODO should we separate the phrase counting and the stem matching?
    conn.execute(r#"
        create table if not exists ngrams (
            label text,
            n integer not null,
            ngram text not null,
            ngram_stemmed text not null,
            left_parent text,
            right_parent text,
            count integer not null,
            document_count integer not null,
            primary key (label, ngram, left_parent, right_parent)
        );
    "#, NO_PARAMS)?;

    conn.execute(r#"
        create table if not exists partition_stats (
            label text,
            head text not null,
            position integer not null,
            primary key (label, head)
        );
    "#, NO_PARAMS)?;

    Ok(conn)
}

#[post("/documents", data = "<documents>")]
fn create_documents(documents: Json<Vec<Document>>) -> String {
    let mut conn = Connection::open(DB_PATH).expect("Couldn't open database conn.");
    for document in &documents.0 {
        if let Err(_err) = save_body_and_labels(&mut conn, &document.body, &document.labels) {
            return format!("err: Unable to save document: {:?}", document);
        }
    }
    String::from("ok")
}

#[get("/documents/statistics/<label>")]
fn get_document_statistics(label: Option<String>) -> String {
    let conn = Connection::open(DB_PATH).expect("Couldn't connect to database");
    let mut stmt = conn.prepare(r#"
        select 
            count(*) as count
        from documents
        where label = ?
    "#).unwrap();
    let count: i64 = stmt.query_row(&[&label], |row| row.get(0)).unwrap();
    format!("Total documents: {}", count)
}


#[get("/hello")]
fn hello() -> String {
    String::from("Hey there!")
}

fn save_body_and_labels(conn: &mut Connection, body: &String, labels: &Option<Vec<String>>) -> rusqlite::Result<()> {
    if let Some(labels) = labels {
        for label in labels {
            save_body_and_label(conn, body, &Some(label))?;
        }
    } else {
        save_body_and_label(conn, body, &None)?;
    }
    Ok(())
}

fn save_body_and_label(conn: &mut Connection, body: &String, label: &Option<&String>) -> rusqlite::Result<()> {
    // let tx = conn.transaction()?;
    // tx.execute(r#"insert into documents (body, label) values (?, ?)"#, params![body, label])?;
    // let position = tx.last_insert_rowid();
    // tx.execute(r#"
    //     insert or replace into partition_stats (label, position, head) values ($1, $2, 'write');
    // "#, params!(label, &position)).unwrap();
    // tx.commit()?;
    conn.execute(r#"insert into documents (body, label) values (?, ?)"#, params![body, label])?;
    Ok(())
}


fn serve(port: u16) {
    let config = rocket::Config::build(rocket::config::Environment::Development)
        .port(port).finalize().expect("Couldn't create config.");
    rocket::custom(config)
        .mount("/", routes![hello, create_documents, get_document_statistics])
        .launch();
}

fn update_phrase_models(labels: Option<Vec<String>>) -> rusqlite::Result<()> {
    if let Some(labels) = labels {
        for label in labels {
            update_phrase_model(Some(&label))?;
        }
    } else {
        update_phrase_model(None)?;
    }
    Ok(())
}

fn count_ngrams(body: &String, ngrams: &mut HashMap<usize, HashMap<VecDeque<String>, u64>>) {
    // info!("Counting ngrams!");
    let re = Regex::new("[\\w']+").unwrap();
    let mut token_queues: Vec<VecDeque<String>> = Vec::new();
    for i in 1..MAX_NGRAM {
        token_queues.push(VecDeque::with_capacity(i));
    }

    for token in re.find_iter(body) {
        // if let Some(token) = token {
            // info!("Processing token {}", token.as_str());
            let token_string = token.as_str().to_string();
            for i in 1..MAX_NGRAM {
                if let Some(queue) = token_queues.get_mut(i - 1) {
                    queue.push_back(token_string.to_owned());
                    if queue.len() > i {
                        queue.pop_front();
                    }
                    if queue.len() == i {
                        if let Some(ngrams) = ngrams.get_mut(&i) {
                            if let Some(count) = ngrams.get_mut(&queue) {
                                *count += 1;
                            } else {
                                ngrams.insert(queue.to_owned(), 1);
                            }
                        }
                    }
                }
            }
        // }
    }

    // info!("Counted stuff: {:?}", ngrams);
}

fn ensure_read_head(conn: &mut Connection, label: Option<&String>) -> rusqlite::Result<usize> {
    conn.execute(r#"
        insert or replace into partition_stats (label, position, head) values (
            $1,
            coalesce((select position from partition_stats where label = $1 and head = 'read'), 0),
            'read'
        )
    "#, params!(label))
}

fn update_phrase_model(label: Option<&String>) -> rusqlite::Result<()> {
    info!("Updating phrase model for label {:?}.", &label);

    let mut conn = Connection::open(DB_PATH)?;
    let mut ngrams: HashMap<usize, HashMap<VecDeque<String>, u64>> = HashMap::new();

    for i in 1..MAX_NGRAM {
        ngrams.insert(i, HashMap::new());
    }

    info!("{:?}", label);

    ensure_read_head(&mut conn, label)?;

    let _res : Vec<rusqlite::Result<()>> = conn.prepare(r#"
        select body from documents where label = $1 and rowid > (
            select position from partition_stats where head = 'read' and label = $1
        );
    "#)?.query_map(params!(label), |row| {
        Ok(count_ngrams(&row.get(0)?, &mut ngrams))
    })?.collect();

    for i in 1..MAX_NGRAM {
        println!("{} len: {}", i, ngrams.get(&i).unwrap().len());
    }

    Ok(())
}


fn analyze_stdin(labels: Option<Vec<String>>) {
    let mut conn = Connection::open(DB_PATH).expect("Unable to connect to database.");
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        match stdin.read_line(&mut line) {
            Err(err) => {
                error!("error: {}", err);
                std::process::exit(1);
            },
            Ok(bytes) => {
                if bytes == 0 {
                    break;
                } else {
                    save_body_and_labels(&mut conn, &line, &labels).expect("Couldn't save line.");
                }
            }
        }
    }

    update_phrase_models(labels).expect("Failed to update phrase models");
}

fn analyze_file(path: &str, labels: Option<Vec<String>>) {
    let file = std::fs::File::open(path).unwrap();
    let mut conn = Connection::open(DB_PATH).expect("Unable to connect to database.");
    for line in BufReader::new(file).lines() {
        let line = line.expect("Couldn't read line from file.");
        save_body_and_labels(&mut conn, &line, &labels).expect("Couldn't save line.");
    }

    update_phrase_models(labels).expect("Failed to update phrase models");
}

fn cmd_analyze(path: &str, labels: Option<Vec<String>>) {
    if path == "-" {
        analyze_stdin(labels);
    } else {
        analyze_file(path, labels);
    };
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
        )
        (@subcommand serve =>
            (about: "Start API server")
            (@arg port: -p --port +takes_value "Port to serve on (default 6220)")
        )
    ).get_matches();

    setup_database().expect("Could not perform initial DB setup.");

    if let Some(matches) = matches.subcommand_matches("serve") {
        let port = matches.value_of("port").unwrap_or("6220").parse::<u16>().expect("Couldn't parse port.");
        serve(port);
    } else if let Some(matches) = matches.subcommand_matches("analyze") {
        env_logger::init();
        info!("Hello.");
        let labels = matches.values_of("label").map(|values| values.map(String::from).collect());
        match matches.value_of("input") {
            Some(path) => {
                cmd_analyze(path, labels);
            },
            None => {
                error!("Must provide a file to read text from, or pass - and stream to stdin.");
                std::process::exit(1);
            }
        }
    }
    
}
