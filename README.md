# CAST
Code and dataset of our paper "[Context-Aware Seq2seq Translation Model for Sequential Recommendation]" submitted to Information Sciences.

## 1. Requirements
* python 3.6.2
* pytorch 1.6.0
* numpy 1.18.1

## 2. Usage
 To train our model on MovieLens (with default hyper-parameters):
```
sh run_ml1m.sh
```
 Note that ```test_random_examples_500_XXXX.dat``` is a dict containing all negative samples for validation or test, and all baselines use the same negative samples for a fair comparison. 
 Result logs and models on each datasets are also provided.
 As for the JD and Taobao, the files are too big to upload. Please download the files on the following links:
* JD: [Click here.](https://drive.google.com/file/d/1VauneB1bS6N3kDFWO_Xh2WM--6hHOmsA/view?usp=sharing)
* Taobao: [Click here.](https://drive.google.com/file/d/12_p_RnHdHMh5GDKbi40lCkNO-FKxKjAA/view?usp=sharing)
