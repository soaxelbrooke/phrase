
# Phrase

A tool for learning significant phrase/term models, and efficiently labeling with them.

## Use

### Installation

**TODO**

### Training a phrase model

First, you need to count ngrams from your data:

```
$ phrase count assets/reviews.csv --csv
```

(This creates ngram count files under `data/`)

Then, you need to export scored phrase models:

```
$ phrase export
```

(This will create scored phrase models under `data/`)

### Serving Scored Prhase Models

```
$ phrase serve
```

```
$ curl -XPOST localhost:6220/analyze -d '{"documents": [{"label": "6001", "body": "The weather channel is great for when I want to check the weather!"}]}'
[{"body":"The weather channel is great for when I want to check the weather!","label":"6001","ngrams":["channel","check the weather","weather","weather channel"]}]
```
