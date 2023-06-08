# EMBERSim: A Large-Scale Databank for Boosting Similarity Search in Malware Analysis

Data is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8014709.svg)](https://doi.org/10.5281/zenodo.8014709)
- EMBER metadata with AVClass2 re-run
- AVClass2 tag co-occurrence results
- XGBoost leaf similarity results

## Usage

First, refer to EMBER repo (https://github.com/elastic/ember) for instructions on how to obtain train & test datasets, as well as how to run feature extraction.

### Train the model

With `src/xgboost_trainer.py` you can:
- perform a grid search to find the best hyperparameters values for the xgboost model
- train an xgboost model with the hyperparameters values recommended by us
- train an xgboost model with your own hyperparameters values
Check the source for details about the required arguments.

### Compute the leaf predictions

With `src/leaf_similarity/leaf_pred_predictions.py` you can:
- generate the leaf predictions dataset for train and test starting from an xgboost model and the ember dataset
- generate the prediction scores for the unlabelled subset from the EMBER dataset
Check the source for details about the required arguments.

### Get the top 100 most similar entries for any sample

This is a time consuming task. As so, we devided the process into two steps:

1. With `src/leaf_similarity/leaf_pred_top_100_search.py` you can:
- generate the top 100 similar hits for different `query` vs `knowledge base` combinations, such as: test vs train + test, unlabelled vs train, unlabelled vs train + test
- this will save the results in multiple pickle files
Check the source for details about the required arguments.

1. With `src/leaf_similarity/leaf_pred_top_100_shas.py` you can, starting from the results in the previous step:
- generate a csv file where for a sha there will be 100 most similar other shas, depending on the targeted datasets
- you can also specify if the similarity score should be present or not in the results
Check the source for details about the required arguments.

### Get similarity search statistics

You can compute the statistics of the similarity search both from the binary labels and multiclass labels.

- With `src/leaf_similarity/leaf_pred_binary_stats.py`, you can get statistics of the leaf similarity search based on the benign / malicious labels
- With `src/leaf_similarity/leaf_pred_class_stats.py`, you can get statistics of the leaf similarity search based on the class presented in the EMBER metadata
Check the source for details about the required arguments.

### Running AVClass

First, ensure you cloned the AVClass repo (https://github.com/malicialab/avclass).

Given a JSONL input file (one JSON object per line) with VirusTotal detection results, you can use the `src/run_avclass.sh` script to run the AVClass' labeler for obtaining sample tags.

Tag-related operations (e.g. augmentation via co-occurrence, ranking etc.) are defined in `src/dataset.py`.

#### Parse AVClass results and add to EMBER dataframe
You can parse the AVClass results and augment a dataframe with original EMBER metadata by using `src/parse_avclass.py`.
Example:
```
python3 parse_avclass.py \
  --avclass-results-file avclass_results.txt \
  --ember-dataframe-csv ember_original_metadata.csv \
  --output-dataframe-path ember_with_avclass_dataset.csv
```
This dataset is already provided, see DOI at the beginning of this README.

#### Adding tags via co-occurrence
Given a dataframe with EMBER metadata, AVClass tag co-occurrence information (AVClass `.alias` file) and a co-occurrence threshold,
you can use `TagAugmenter` from `src/dataset.py` to add extra tags to samples:
```python
tag_assoc = TagAssociations("avclass_tag_co_occurrence.alias")
tag_aug = TagAugmenter(tag_assoc, thr_co_occur=0.9)
dataframe = pd.read_csv("ember_with_avclass_dataset.csv")
dataframe["EXTRA_TAGS"] = dataframe.apply(tag_aug.resolve_final_tags, axis=1)
```

#### Ranking tags
To obtain tag rankings (for `FAM` or `CLASS` tags, e.g. in order to prepare for evaluation), you can use `get_tag_ranking` from `src/dataset.py`:
```python
dataframe = pd.read_csv("ember_with_avclass_dataset.csv")
# computing ranks requires having non-null AVClass tag info and co-occurrence info (from the step above)
dataframe["TAG_RANKS"] = dataframe.query("avclass_curr.notna() & EXTRA_TAGS.notna()").apply(
  lambda row: get_tag_ranking(
      tag_scores=row["avclass_curr"],
      co_occurrence=row["EXTRA"],
      tag_kind="FAM", # rank by FAM tags
      return_scores=False,
  ),
  axis=1,
)
```

### Evaluation

For evaluating the XGBoost-based similarity search, you can start from the `src/e2e.py` script, which does the following:
- loads the original EMBER dataset and constructs `TAG_RANKS` for either `CLASS` or `FAM` tags
  - for ranking tags using tag co-occurrence information, the `.alias` file is required, as constructed by AVClass
  - if you wish to not use tag co-occurrence information, there is an option to keep only the most prevalent tag (i.e. by AVClass rank score) as the ground truth for a sample
- loads the similarity search results from a dataframe structured as: `needle_sha256 -> [hits_sha256]`
  - for info on how to generate top-N most similar samples for a given query, see above
- computes relevance@K using a relevance function specified by the user: exact match, IOU (intersection over union), normalized edit similarity

After results are dumped to the pickle file, you can explore them using the `notebooks/evaluation.ipynb` notebook, which provides functionality for plotting histograms, empirical CDFs and constructing a summary table with descriptive statistics.

Evaluation metrics are implemented in `src/evaluation.py`.

## References

- Original EMBER repo: https://github.com/elastic/ember
- AVClass: https://github.com/malicialab/avclass
- VirusTotal: https://virustotal.com

## Support statement

EMBERSim is an open source project, not a CrowdStrike product. As such, it carries no formal support, expressed or implied.