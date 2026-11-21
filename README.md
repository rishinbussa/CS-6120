# CS 6120 Natural Language Processing coursework

Notebooks from CS 6120 Natural Language Processing at Northeastern University (Fall 2025). Each one is a course assignment: the course provided the starter code and data, and the parts left for students to complete hold my solutions and analysis. All results below are copied from the outputs saved in the notebooks.

| Notebook | Topic | Headline result |
|---|---|---|
| [`token_counts.ipynb`](token_counts.ipynb) | Zipf's law and Shannon entropy | 69,635,098 tokens and 333,518 word types counted; "the" has relative frequency 0.064170 |
| [`reviews.ipynb`](reviews.ipynb) | Movie review sentiment with sparse features | 0.8320 mean 10-fold accuracy (word counts, L2 logistic regression) |
| [`bert_probe.ipynb`](bert_probe.ipynb) | Probing BERT layers for named entities | 0.9725 word accuracy at layer 11 vs. 0.7650 for always predicting `O` |
| [`cross_language.ipynb`](cross_language.ipynb) | Cross-language retrieval on Wikipedia | Ensemble raises Chinese-to-English recall@1 from 0.717 to 0.804 |

## 1. Zipf's law and Shannon entropy (`token_counts.ipynb`)

**Question.** How predictable is English text to a human reader, and how skewed is the distribution of words in a large corpus?

**Data.**
- My guess counts from the course's online Shannon guessing game for three short texts.
- A sample of 1,000 English books from Project Gutenberg provided by the course (`pg-sample.json.gz`, 144 MB compressed).

**Method.**
- Arranged the guesses into a table of context length by number of guesses, then computed Shannon's upper bound (entropy of the guess distribution) and lower bound for each context length.
- Tokenized the books by lowercasing and keeping runs of letters (`[a-z]+`), counted unigrams, bigrams and trigrams, and plotted rank against relative frequency on log-log axes with a slope -1 reference line.

**Results.**
- 69,635,098 tokens and 333,518 word types.
- Most frequent words: "the" (4,468,492; relative frequency 0.064170), "of" (2,486,678; 0.035710), "and" (2,286,006; 0.032828).
- Most frequent bigram: "of the" (663,196). Most frequent trigram: "one of the" (21,817).
- Unigram frequencies run roughly parallel to the slope -1 line through the middle ranks and fall off faster in the rare-word tail. The bigram and trigram curves are flatter than slope -1.
- With no context, the guessing-game bounds are 1.5850 bits (upper) and 0.0884 bits (lower). Each context length has at most three observations (one per text), so the upper bound can only take a few values (0, 0.9183, 1.0000 or 1.5850 bits) and does not fall steadily as context grows.

**Next steps.** Pool guesses from more texts and group context lengths into bins so each estimate rests on enough data. Fit the Zipf exponent by regression on log rank and log frequency instead of judging the plot by eye. Handle apostrophes, since the letters-only rule splits "don't" into "don" and "t".

## 2. Sentiment classification with sparse features (`reviews.ipynb`)

**Question.** How much does the choice of document features matter when classifying movie reviews as positive or negative with logistic regression?

**Data.** The Cornell movie review polarity corpus as distributed by the course: 2,000 reviews, 1,000 positive and 1,000 negative.

**Method.**
- Basic features: counts of alphabetic words (`CountVectorizer`, token pattern `[a-zA-Z]+`).
- Extended features: TF-IDF weighted unigrams and bigrams (`TfidfVectorizer`, `ngram_range=(1, 2)`).
- Logistic regression (liblinear solver) with L1 and with L2 regularization, scored by mean accuracy over shuffled 10-fold cross-validation. The 20 largest positive and negative weights of a model fit on all reviews give the most indicative terms.

**Results** (mean 10-fold accuracy):

| Features | L1 | L2 |
|---|---|---|
| Word counts | 0.8265 | 0.8320 |
| TF-IDF unigrams and bigrams | 0.7035 | 0.8060 |

- Word counts with L2 scored best. Top positive terms include "fun", "great", "excellent", "memorable" and "terrific"; top negative terms include "bad", "unfortunately", "worst", "waste" and "boring".
- TF-IDF features scored lower under both penalties. With L1 they kept so few nonzero weights that the same zero-weight terms ("predicts", "predictions will") appear at the end of both the positive and the negative top-20 lists.

**Next steps.** Tune the regularization strength for each feature set with nested cross-validation (every model here used the default `C=1.0`). Fit the vectorizers inside each fold with a scikit-learn `Pipeline` so the vocabulary and IDF weights come only from training data. Add negation handling and compare against a small fine-tuned transformer.

## 3. Probing BERT for named entities (`bert_probe.ipynb`)

**Question.** How much named-entity information can a linear classifier read from each layer of BERT, a model that never saw entity labels in pretraining?

**Data.** The first 1,000 training sentences of CoNLL-2003 English NER (`hgissbkh/conll2003-en` on the Hugging Face Hub): 12,057 words, each tagged with one of nine BIO labels (B- and I- tags for person, organization, location and miscellaneous, plus `O`).

**Model.** `bert-base-cased`, using the hidden states of all 13 layers (input embeddings plus 12 transformer layers, 768 dimensions each). Each word is represented by its first subword token.

**Method.** One logistic regression probe per layer, scored by word-level accuracy averaged over shuffled 10-fold cross-validation and compared with always predicting `O`. A second set of probes predicts whether a word starts with a capital letter.

**Results** (mean 10-fold accuracy):
- NER: 0.9250 at layer 0, rising to 0.9722 at layer 6 and staying between 0.9716 and 0.9725 through layer 11 (best: layer 11, 0.9725), then 0.9700 at layer 12. Always predicting `O` scores 0.7650.
- Capitalization: 1.0000 at layers 0 to 2, falling slowly to 0.9974 at layer 12. Always predicting lowercase scores 0.6985.
- Case is fully recoverable from the input embeddings, as expected for a cased vocabulary, and fades slightly with depth, while entity information keeps improving into the middle layers.

**Next steps.** Split folds by sentence and add a control task (Hewitt and Liang, 2019): with word-level folds a probe can memorize frequent entity words, and the input embedding layer alone already reaches 0.9250. Report entity-level F1 with `seqeval` alongside word accuracy. Compare against a fine-tuned token classifier.

## 4. Cross-language retrieval on Wikipedia (`cross_language.ipynb`)

**Question.** Given the first paragraph of a Wikipedia article in Arabic, German, Greek, French or Chinese, can a multilingual sentence encoder find the English article on the same topic?

**Data.** 11,838 first paragraphs of Wikipedia articles in six languages (English, German, French, Chinese, Arabic, Greek), linked across languages by Wikidata ID (`sample-6lang.jsonl`, 7.2 MB, provided by the course). Linked paragraphs cover the same topic but are mostly not translations of each other.

**Models.** `sentence-transformers/LaBSE` and `paraphrase-multilingual-mpnet-base-v2`.

**Method.** For each language, encode the first 1,000 paragraphs as queries and the first 1,000 English paragraphs as candidates, rank candidates by cosine similarity, and measure recall@k: the share of queries whose linked English article is in the top k. The improved method averages the similarity matrices of the two models with equal weights.

**Results** (recall@1 / recall@5 / recall@10, 1,000 queries per language):

| Query language | LaBSE | LaBSE + MPNet ensemble | Change in recall@1 |
|---|---|---|---|
| Arabic | 0.868 / 0.945 / 0.959 | 0.909 / 0.962 / 0.971 | +0.041 |
| German | 0.885 / 0.949 / 0.962 | 0.956 / 0.974 / 0.981 | +0.071 |
| Greek | 0.884 / 0.948 / 0.957 | 0.939 / 0.967 / 0.971 | +0.055 |
| French | 0.856 / 0.929 / 0.944 | 0.935 / 0.963 / 0.970 | +0.079 |
| Chinese | 0.717 / 0.831 / 0.876 | 0.804 / 0.899 / 0.930 | +0.087 |

The ensemble improved every language at every k, with the largest recall@1 gain for Chinese, the weakest language under LaBSE alone.

**Next steps.** Choose the ensemble weight on paragraphs outside the evaluation set instead of fixing it at 0.5. Fine-tune LaBSE on linked article pairs with an in-batch contrastive loss. Rerank the top 10 candidates with a cross-encoder, since the ensemble's recall@10 is already 0.930 to 0.981.

## Run it

**Colab.** The first cell of each notebook has an "Open in Colab" link. `bert_probe.ipynb` and `cross_language.ipynb` were run on a Colab T4 GPU runtime; the other two need no GPU.

**Local Jupyter.**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook
```

`requirements.txt` lists only the third-party packages the notebooks import. Each notebook downloads its own data when run:

| Notebook | Downloads |
|---|---|
| `token_counts.ipynb` | `pg-sample.json.gz` (144 MB) from the course website, via `wget` |
| `reviews.ipynb` | `cornell_reviews.json` from the course's GitHub repository, via `requests` |
| `bert_probe.ipynb` | `hgissbkh/conll2003-en` and `bert-base-cased` from the Hugging Face Hub |
| `cross_language.ipynb` | `sample-6lang.jsonl` from the course's GitHub repository via `wget`, plus LaBSE (1.88 GB) and multilingual MPNet from the Hugging Face Hub |

If `wget` is not installed, download the file from the URL in that cell into the notebook's folder. `token_counts.ipynb` holds all 69,635,098 tokens plus the bigram and trigram counts in memory, so it needs a machine with plenty of RAM.
