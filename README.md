
<p align="center">
  <img width="256" height="256" title="phrase" src="https://user-images.githubusercontent.com/2815794/57149171-faf2c880-6d7f-11e9-901f-3010f9abc443.png">
</p>

A CLI tool and server for learning significant phrase/term models, and efficiently labeling with them.

## Installation

Download and extract the [release archive](https://github.com/soaxelbrooke/phrase/releases) for your OS, and put the `phrase` binary somewhere on the PATH (like `/usr/local/bin`).  If you're using linux, the GNU binary currently appears to be 5-10x faster than the musl version, so try that first.

For example, installing the linux binary:

```
$ wget https://github.com/soaxelbrooke/phrase/releases/download/0.3.5/phrase-0.3.5-x86_64-unknown-linux-gnu.tar.gz
$ tar -xzvf phrase-0.3.5-x86_64-unknown-linux-gnu.tar.gz
$ sudo mv phrase /usr/local/bin/
```

## Use

In general, using `phrase` falls into 3 steps:

1. Counting n-grams
2. Exporting scored models
3. Significant term/phrase extraction/transform or model serving

N-gram counting is done continuously, providing batches of documents as they come in. Model export reads all n-gram counts so far and calculates mutual information-based collocations - you can then deploy the models by shipping the binary and `data/scores_*` files to a server.  Labeling (identifying all significant terms and phrases in text) or transforming (eager replace of longest found phrases in text) can be done either via the CLI or the web server. [Providing labels](#labels) for documents is not necessary for learning phrases, but does help, and allows for significant term labeling also.

### Training a phrase model

This example uses the `assets/reviews.json` data in the repo, 10k app reviews:

```
$ head -1 assets/reviews.json
{"body": "Woww! Moon Invoice is just so amazing. I don\u2019t think any such app exists that works so wonderfully. I am awestruck by the experience.", "category": "Business", "sentiment": "positive"}
```

First, you need to count n-grams from your data:

```
$ phrase count --mode json assets/reviews.json --textfield body --labelfield category --labelfield sentiment
```

(This creates n-gram count files at `data/counts_*`)

Then, you need to export scored phrase models:

```
$ phrase export
```

(This will create scored phrase models at `data/scores_*`)

### Validating Learned Phrases

You can validate the phrases being learned per-label with the `show` command:

```
$ phrase show -n 3

Label=News
hash,ngram,score
3178089391134982486,New Yorker,0.5142287028163096
18070968419002659619,long form,0.5096737783425647
16180697492236521925,sleep timer,0.5047391214969927

Label=Business
hash,ngram,score
4727477585106156155,iTimePunch Plus,0.55749203444841
12483914742025992948,Crew Lounge,0.5479129370086021
11796198430323558093,black and white,0.5385891753319711

...
```

`hash` is the hash of the stemmed n-gram, `ngram` is the canonical version of the n-gram used for display purposes.  For phrases, `score` is a combination of NPMI(phrase, tokens) and NPMI(n-gram, label), and is only NPMI(n-gram, label) for single tokens.

### Transforming Text

```
$ echo "The weather channel is great for when I want to check the weather!" | phrase transform --label Weather -
The Weather_Channel is great for when I_want_to check_the_weather!
```

Modes allow CSV, JSON, and plaintext (the default). CSV and JSON will maintain the rest of the document/row, but replace text in the specified `--textlabel` fields (or in the `text` field if not specified).

### Serving Scored Phrase Models

```
$ phrase serve
```

It also accepts `--port` and `--host` parameters.

### API Routes

**GET /labels** - list all available labels for extraction/labeling.

```
$ curl localhost:6220/labels
{"labels":["Social Networking","Travel","negative","Weather","positive","Business","News","neutral",null]}
```

**POST /analyze** - identifies all significant phrases and terms found in the provided documents.

```
$ curl -XPOST localhost:6220/analyze -d '{"documents": [{"labels": ["Weather", "positive", null], "text": "The weather channel is great for when I want to check the weather!"}]}'
[{"labels":["Weather","positive"],"ngrams":["I want","I want to","I want to check","Weather Channel","channel","check","check the weather","want to","want to check","want to check the weather","weather","when I want","when I want to"],"text":"The weather channel is great for when I want to check the weather!"}]
```

**POST /transform** - eagerly replaces the longest phrases found in the provided documents.

```
$ curl -XPOST localhost:6220/transform -d '{"documents": [{"labels": ["Weather"], "text": "The weather channel is great for when I want to check the weather!"}]}'
[{"label":"Weather","text":"The Weather_Channel is great for when I_want_to check_the_weather!"}]
```

## Labels

Labels are used to learn significant single tokens and to aid in scoring significant phrases.  While `phrase` can be used without providing labels, providing them allows it to learn more nuanced phrases, like used by a specific community or when describing a specific product.  Labels are generally provided in the `label` field of the input file, specified using `--labelfield` argument, or with the `--label` argument.

Providing labels for your data causes `phrase` to count them into separate bags per label, and during export allows it to calculate an extra significance score based on label (instead of just co-occurance).  This means that a phrase that is unique to that label is much more likely to be picked up than if it was being overshadowed in unlabeled data.

An example of a good label would be app category, as apps in each category are related, and customer reviews talk about similar subjects.  An example of a bad label would be user ID, since it would be very high cardinality, cause very bad performance, and likely wouldn't learn useful phrases or terms due to data sparsity per user.

## Performance

It's fast.

It takes 0.66 second to count 1 to 5-grams for 10,000 reviews, and ~1.2 seconds to export. Performance is primarily based on n-gram size, the number of labels, and vocab size.  For example, labeling on iOS app category (23 labels) using default parameters on an Intel Core i7-7820HQ (Ubuntu):

|Task|Tokens per Second per Thread|
|----|----------------------------|
|Counting n-grams|779,025|
|Exporting scored models|206,704|
|Labeling significant terms|354,395|
|Phrase transformation|345,957|

_Note:_ Exports do not gain much from parallelization

## Environment Variables

A variety of environment variables can be used:

`LANG` - Determines the stemmer language to use, [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).  Should be set automatically on Unix systems, but can be overridden.

`TOKEN_REGEX` - The regular expression used to find tokens when learning and labeling phrases.

`CHUNK_SPLIT_REGEX` - The regular expression used to detect chunk boundaries, across which phrases aren't learned.

`HEAD_IGNORES` / `TAIL_IGNORES` - Used to ignore phrases that start or end with a token, comma separated.  For instance, `TAIL_IGNORES=the` would ignore 'I love the'.

`PRUNE_AT` - The size at which to prune the n=gram count mapping.  Useful for limiting memory usage, default is 5000000.

`PRUNE_TO` - Controls what size n-gram mappings are pruned to during pruning.  Also sets the number of n-grams that are saved after counting (sorted by count). Default is 2000000.

`BATCH_SIZE` - Controls the document batch size.  Causes input streams to be batched, allowing larger than memory datasets.  Default is 1000000.

`MAX_NGRAM` - The highest n-gram size to count to, higher values cause slower counting, but allow for more specific and longer phrases. Default is 5.

`MIN_NGRAM` - The lowest n-gram size to export, default is 1 (unigrams).

`MIN_COUNT` - The minimum n-gram count for a phrase or token to be considered significant.  Default is 5.

`MIN_SCORE` - The minimum NPMI score for a term or phrase to be considered significant.  Default is 0.1.

`MAX_EXPORT` - The maximum size of exported models, per label.

`NGRAM_DELIM` - The delimiter used to join phrases when counting and scoring.  Default is ` `.

## Citations

[Normalized (Pointwise) Mutual Information in Collocation Extraction - Gerlof Bouma](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
