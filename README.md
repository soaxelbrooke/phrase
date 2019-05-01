
# Phrase

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
stemmed,ngram,score
itimepunch plus,iTimePunch Plus,0.5478097727335406
black and white,black and white,0.5301331992513779
caller id,caller ID,0.5289939800990554

Label=6005
stemmed,ngram,score
seek arrang,seeking arrangement,0.5393761374868443
older women,older women,0.5387631792730164
younger men,younger men,0.5276998455446196

Label=6001
stemmed,ngram,score
lil bub,Lil Bub,0.5918180152617087
dark sky,Dark Sky,0.549581815673372
zip code,zip code,0.5164957628490037

Label=6003
stemmed,ngram,score
road to hana,road to Hana,0.595772467438227
happiest place on earth,happiest place on earth,0.5660341461243589
north shore,North Shore,0.5590151652161709

Label=6009
stemmed,ngram,score
new yorker,New Yorker,0.510585395386124
sleep timer,sleep timer,0.49587662290496715
deal breaker,deal breaker,0.48809862161808737
```

### Transforming Text

```
$ echo 'Quality, choices, and time saving! Moon Invoice has it all.' | phrase transform --label 6000 -
Quality, choices, and time_saving! Moon_Invoice has it all.
```

### Serving Scored Phrase Models

```
$ phrase serve
```

```
$ curl -XPOST localhost:6220/analyze -d '{"documents": [{"label": "6001", "text": "The weather channel is great for when I want to check the weather!"}]}'
[{"text":"The weather channel is great for when I want to check the weather!","label":"6001","ngrams":["channel","check the weather","weather","weather channel"]}]
```

### Environment Variables

`LANG` - Determines the stemmer language to use from this.  Should be set automatically on Unix systems, but can be overridden.

`TOKEN_REGEX` - The regular expression used to find tokens when learning and labeling phrases.

`CHUNK_SPLIT_REGEX` - The regular expression used to detect chunk boundaries, across which phrases aren't learned.

`HEAD_IGNORES / TAIL_IGNORES` - Used to ignore phrases that start or end with a token, comma separated.  For instance, `TAIL_IGNORES=the` would ignore 'I love the'.

`PRUNE_AT` - The size at which to prune the ngram count mapping.  Useful for limiting memory usage, default is 5000000.

`PRUNE_TO` - Controls what size ngram mappings are pruned to during pruning.  Also sets the number of ngrams that are saved after counting (sorted by count).

`MAX_NGRAM` - The highest ngram size to count to, higher values cause slower counting, but allow for more specific and longer phrases. Default is five.

`MIN_COUNT` - The minimum ngram count for a phrase or token to be considered significant.  Default is 5.

`MIN_SCORE` - The minimum NPMI score for a term or phrase to be considered significant.  Default is 0.1.

`MAX_EXPORT` - The maximum size of exported models, per label.

`NGRAM_DELIM` - The delimiter used to join phrases when using the `transform` subcommand.  Default is _: `fax machine` -> `fax_machine`.

### References

[Normalized (Pointwise) Mutual Information in Collocation Extraction - Gerlof Bouma](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
