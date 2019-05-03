
<p align="center">
  <img width="256" height="256" title="phrase" src="https://user-images.githubusercontent.com/2815794/57115077-811bfa00-6d01-11e9-8f3b-b49be93a35e1.png">
</p>

A tool for learning significant phrase/term models, and efficiently labeling with them.

## Use

### Installation

Download and extract the [release archive](https://github.com/soaxelbrooke/phrase/releases) for your OS, and put the `phrase` binary somewhere on the PATH (like `/usr/local/bin`).

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

### Validating Learned Phrases

```
$ phrase show -n 3

Label=6000
hash,ngram,score
4727477585106156155,iTimePunch Plus,0.55749203444841
12483914742025992948,Crew Lounge,0.5479129370086021
11796198430323558093,black and white,0.5385891753319711

Label=6005
hash,ngram,score
5028985570365810872,seeking arrangement,0.555805973833051
3452512271924928155,older women,0.5492325901167289
1856639309753967090,younger men,0.5380823555018927

Label=6001
hash,ngram,score
16860550243828630957,Lil Bub,0.5476212238386382
15429443734362810908,Dark Sky,0.538136969505567
17787744274931639214,zip code,0.5126091671796332

Label=6003
hash,ngram,score
16827980792972836770,Na Pali,0.6170795527743764
4230405495922241687,road to Hana,0.5618339661289787
8724945449383040970,fast pass,0.5496929588451972

Label=6009
hash,ngram,score
3178089391134982486,New Yorker,0.5142287028163096
18070968419002659619,long form,0.5096737783425647
16180697492236521925,sleep timer,0.5047391214969927
```

### Transforming Text

```
$ echo 'Love the black and white! Moon Invoice has it all.' | phrase transform --label 6000 -
Love the black_and_white! Moon_Invoice has it all.
```

### Serving Scored Phrase Models

```
$ phrase serve
```

```
$ curl -XPOST localhost:6220/analyze -d '{"documents": [{"label": "6001", "text": "The weather channel is great for when I want to check the weather!"}]}'
[{"text":"The weather channel is great for when I want to check the weather!","label":"6001","ngrams":["channel","check the weather","weather","weather channel"]}]
```

## Labels

Labels are used to learn significant single tokens and to aid in scoring significant phrases.  While `phrase` can be used without providing labels, providing them allows it to learn more nuanced phrases, like used by a specific community or when describing a specific product.

Providing labels for your data causes `phrase` to count them into separate bags per label, and during export allows it to calculate an extra significance score based on label (instead of just co-occurance).  This means that a phrase that is unique to that label is much more likely to be picked up than if it was being overshadowed in unlabeled data.

An example of a good label would be app category, as apps in each category are related, and customer reviews talk about similar subjects.

## Environment Variables

A variety of environment variables can be used:

`LANG` - Determines the stemmer language to use from this.  Should be set automatically on Unix systems, but can be overridden.

`TOKEN_REGEX` - The regular expression used to find tokens when learning and labeling phrases.

`CHUNK_SPLIT_REGEX` - The regular expression used to detect chunk boundaries, across which phrases aren't learned.

`HEAD_IGNORES / TAIL_IGNORES` - Used to ignore phrases that start or end with a token, comma separated.  For instance, `TAIL_IGNORES=the` would ignore 'I love the'.

`PRUNE_AT` - The size at which to prune the ngram count mapping.  Useful for limiting memory usage, default is 5000000.

`PRUNE_TO` - Controls what size ngram mappings are pruned to during pruning.  Also sets the number of ngrams that are saved after counting (sorted by count).

`MAX_NGRAM` - The highest ngram size to count to, higher values cause slower counting, but allow for more specific and longer phrases. Default is 5.

`MIN_NGRAM` - The lowest ngram size to export, default is 1 (unigrams).

`MIN_COUNT` - The minimum ngram count for a phrase or token to be considered significant.  Default is 5.

`MIN_SCORE` - The minimum NPMI score for a term or phrase to be considered significant.  Default is 0.1.

`MAX_EXPORT` - The maximum size of exported models, per label.

`NGRAM_DELIM` - The delimiter used to join phrases when using the `transform` subcommand.  Default is _: `fax machine` -> `fax_machine`.

### References

[Normalized (Pointwise) Mutual Information in Collocation Extraction - Gerlof Bouma](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
