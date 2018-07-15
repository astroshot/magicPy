# Deep Learning Algorithms - Python
## Algorithms are implemented based on Python 3

### Training Data Format
All training data input in the format of `pandas.DataFrame`.
For example:
```bash
In [2]: data_train = pd.read_csv('tests/learning_method/tree/test_credit.csv')

In [3]: data_train
Out[3]: 
       Age  Job House     Credit Classification
0    Young   No    No     Normal             No
1    Young   No    No       Good             No
2    Young  Yes    No       Good            Yes
3    Young  Yes   Yes     Normal            Yes
4    Young   No    No     Normal             No
5   Medium   No    No     Normal             No
6   Medium   No    No       Good             No
7   Medium  Yes   Yes       Good            Yes
8   Medium   No   Yes  Very Good            Yes
9   Medium   No   Yes  Very Good            Yes
10     Old   No   Yes  Very Good            Yes
11     Old   No   Yes       Good            Yes
12     Old  Yes    No       Good            Yes
13     Old  Yes    No  Very Good            Yes
14     Old   No    No     Normal             No
```
Column names are used as the name of each feature. **The last Column of training data is result.**
All Models of this project are based on this.

### Decision Tree
ID3 method and C4.5 method are implemented.

### Naive Bayes
Maximum Likelihood and Bayes Method are implemented.

add some text
