# Movie review sentiment classification

## Naive Bayes

Instructions [here](https://www.cl.cam.ac.uk/teaching/1920/L90/Instructions201920.pdf)

Use data-tagged (already tokenized)

Types of classifiers:
1. Unigram
    - No need to do negation tagging
    - Has to appear at least 4 times in the corpus
2. Bigram
    - Has to appear at least 7 times
3. Joint bigram and unigram
    - Use both at the same time
    --> Multiply twice as many probabilities together

-  Train with files cv000–cv899
-  Test with cv900–cv999
-  Use three equal-sized folds while maintaining balanced class distributions in each fold. Report average three-fold cross-validation results
- No need to use stems or stopwords
- Treat punctuation as its own word

## Doc2Vec + SVM

Instructions [here](https://www.cl.cam.ac.uk/teaching/1920/L90/Instructions201920_part2.pdf).

State of the art [here](http://nlpprogress.com/english/sentiment_analysis.html)(~ 95 %).

### Different tasks
- POS vs NEG
- Three bins

### System comparisons using the sign test
- Distributed memory vs. distributed bag of words
- SVM vs. Naive Bayes
- Low dim representation vs. raw doc2vec vectors
- Different SVM kernels

### Hyperparameters
- Vector size
- Epoch amount

### Validation methods for vectors
- Compare mean vector distances for documents in same class
    - Model has the method `distances`
- Plot using t-SNE

### Use cross-validation for system choice