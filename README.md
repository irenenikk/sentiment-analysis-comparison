# Movie review sentiment classification

This project uses python 3.

## Naive Bayes

The relevant code can be found in [the file practical1.py](code/practical1.py).

Original instructions are [here](https://www.cl.cam.ac.uk/teaching/1920/L90/Instructions201920.pdf)

The system includes both a vanilla classifier and one implemented with add-one smoothing. The classifier supports using either unigrams, bigrams, or both as the feature set.

The data used is not included in the repository. You should give the path to the dataset as an argument when running the program:

```
$ python3 code/practical1.py <path to data folder>
```

## SVM

Instructions [here](https://www.cl.cam.ac.uk/teaching/1920/L90/Instructions201920_part2.pdf).

State of the art [here](http://nlpprogress.com/english/sentiment_analysis.html)(~ 95 %).

An expansion to the baseline system where an SVM is used in prediction. Compares systems using BOW-vectors and Doc2Vec
