# EMBERSim: A Large-Scale Databank for Boosting Similarity Search in Malware Analysis

Data available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8014709.svg)](https://doi.org/10.5281/zenodo.8014709)

## Instructions on how to navigate the repository

We assume that you already have cloned the EMBER repo and run their scripts for generating the train and test datasets as well as the metadata file.

### Train the model

In the src/xgboost_trainer.py you can:
- perform a grid search to find the best hyperparameters values for the xgboost model
- train an xgboost model with the hyperparameters values recommended by us
- train an xgboost model with your own hyperparameters values
Please check the file for more details about what parameters the file expects.

### Compute the leaf predictions

In the src/leaf_similarity/leaf_pred_predictions.py you can:
- generate the leaf predictions dataset for train and test starting from an xgboost model and the ember dataset
- generate the prediction scores for the unlabelled subset from the EMBER dataset
Please check the file for more details about what parameters the file expects.

### Get the top 100 similar entries for any sample

This is a time consuming task. As so, we devided the process into two steps:

1. In the leaf_similarity/leaf_pred_top_100_search.py you can:
- generate the top 100 similar hits for different datasets, such as test vs train + test, unlabelled vs train, or unlabelled vs train + test
- this will save the results in multiple pickles
Please check the file for more details about what parameters the file expects.

2. In the leaf_similarityt/leaf_pred_top_100_shas.py you can, starting from the results in the previous step:
- generate a csv file where for a sha there will be 100 most similar other shas, depending on the targeted datasets
- you can also specify if the similarity score should be present or not in the results
Please check the file for more details about what parameters the file expects.

### Get similarity search statistics

One can compute the statistics of the similarity search both from the binary labels and multiclass labels.

In leaf_similarity/leaf_pred_binary_stats.py you can get statistics of the leaf similarity search based on the clean / malicious labels.
Please check the file for more details about what parameters the file expects.

In leaf_similarity/leaf_pred_class_stats.py you can get statistics of the leaf similarity search based on the class presented in the EMBER metadata.
Please check the file for more details about what parameters the file expects.

### Running AVClass

Given a JSONL input file (one JSON object per line) with VirusTotal detection results, you can use the `run_avclass.sh` script to run the AVClass' labeler for obtaining sample tags.

Tag-related operations (e.g. augmentation via co-occurrence, ranking etc.) are defined in `dataset.py`.

#### Parse AVClass results and add to EMBER dataframe
You can parse the AVClass results and augment a dataframe with original EMBER metadata by using `parse_avclass.py`.
Example:
```
python3 parse_avclass.py \
  --avclass-results-file avclass_results.txt \
  --ember-dataframe-csv ember_original_metadata.csv \
  --output-dataframe-path ember_with_avclass.pickle
```

#### Adding tags via co-occurrence
Given a dataframe with EMBER metadata, co-occurrence information (AVClass' `.alias` file) and a co-occurrence threshold,
you can use `TagAugmenter` from `dataset.py` to add extra tags to samples:
```python
tag_assoc = TagAssociations(assoc_file)
tag_aug = TagAugmenter(tag_assoc, thr_co_occur=thr_co_occur)
dataframe["EXTRA_TAGS"] = dataframe.apply(tag_aug.resolve_final_tags, axis=1)
```

#### Ranking tags
To rank `FAM` or `CLASS` tags (e.g. in order to prepare for evaluation), you can use `get_tag_ranking` from `dataset.py`:
```python
# computing ranks requires having non-null AVClass tag info and co-occurrence info
dataframe["TAG_RANKS"] = dataframe.query("avclass_curr.notna() & EXTRA_TAGS.notna()").apply(
  lambda row: get_tag_ranking(
      tag_scores=row["avclass_curr"],
      co_occurrence=row["EXTRA"],
      tag_kind=args.rank_by,
      return_scores=False,
  ),
  axis=1,
)
```

### Evaluation

For evaluating an XGBoost-based similarity search, you can start from the `e2e.py` script, which does the following:
- loads the original EMBER dataset and constructs `TAG_RANKS` for either `CLASS` or `FAM` tags
  - for ranking tags using tag co-occurrence information, the `.alias` file is required, as constructed by AVClass
  - if you wish to not use tag co-occurrence information, there is an option to keep only the most prevalent tag (i.e. with most VT vendor votes) as the ground truth for a sample
- loads the similarity search results from a dataframe structured as: `query -> [hits]`
  - for info on how to generate top-N most similar samples for a given query, see above
- computes relevance@K using a relevance function specified by the user: exact match, edit distance, IOU (intersection over union)

After results are dumped to the pickle file, you can explore them using the `evaluation.ipynb` notebook, which provides functionality for plotting histograms, empirical CDFs and a summary table with descriptive statistics.

Evaluation metrics are implemented in `evaluation.py`.

## References

- Original EMBER repo: https://github.com/elastic/ember
- AVClass: https://github.com/malicialab/avclass
- VirusTotal: https://www.virustotal.com/