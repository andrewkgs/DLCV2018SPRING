# Task 1

## Task Description
Train a classification model on a small amount of data.

## Execution Commands
* Reproduce the results
```sh
bash run.sh <testing data directory> <prediction file path>
```
* Train
```
python3 task_1.py train -tr=<training data directory> -s=<save model directory> --batch_size=<training batch size> --epochs=<training epochs> --valid_ratio=<ratio of validation data>
```
* Test
```
python3 task_1.py test -te=<testing data directory> -l=<load model file path> -p=<prediction file path>
```
