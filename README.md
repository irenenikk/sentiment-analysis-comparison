# Movie review sentiment classification

This is the code used for the practical in the moduel Overview of NLP in Michaelmas 2019. You can find the different tasks in their respective files, named `practical1.py` and `practical2.py`.

Note that this project uses python 3.5.

## Naive Bayes

The relevant code can be found in [the file practical1.py](code/practical1.py).

Original instructions are [here](https://www.cl.cam.ac.uk/teaching/1920/L90/Instructions201920.pdf).

The system includes both a vanilla classifier and one implemented with add-one smoothing. The classifier supports using either unigrams, bigrams, or both as the feature set. The main method will run a series of experiments and print out the results.

The data used is included in the repository. However, you should still give the path to the dataset as an argument when running the program.

```
$ python3 code/practical1.py <path to data folder>
```

## SVM

The relevant code can be found in [the file practical2.py](code/practical2.py).

Instructions [here](https://www.cl.cam.ac.uk/teaching/1920/L90/Instructions201920_part2.pdf).

An expansion to the baseline system where an SVM is used in prediction. Compares systems using BOW-vectors and Doc2Vec.

