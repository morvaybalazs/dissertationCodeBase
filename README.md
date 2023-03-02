# dissertationCodeBase
## steps to be done
### Indexining argument corpus x

### loading topics (read the xml topics task1 only titles 2021 file get topic id and topic history) load it and try to extract topic id topic history COMMENT: Not sure what is topic History Id's are extracted both for 2022 and 20221 X

### retrieving 1000 arguments using Language model wuth dirichlet smoothing (LMDS) or BM25 for each topics and store in a file DONE for 21 and 22 X

### preparing a run file for language model (bm25) retrieval (how to do it: 2021 task1 runs ) X

### evaluate the run file I can use the quality or relevence file that can be downloaded using relevance judgments 

### data representation (feature exteaction) e.g., BERT, Word2Vec, Traditional features(Tf-idf, bm25, LMDS), Linguistic features, Sentiment, Sarcasm, etc.

### training a machine learned ranking model (aka, learning to rank) e.g., SVM-rank or Randomforest

### re-ranking 

### preparing a run file for reranked results

### evaluate the reranked run
