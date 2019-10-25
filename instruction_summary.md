# TODOs

Use data-tagged (tokenized)

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
-  Using folds:
    Use three equal-sized folds while maintaining balanced class distributions in each fold. Report average three-fold cross-validation results
- No need to use stems or stopwords
- Should the words be lowecased?
- Treat punctuation as its own word