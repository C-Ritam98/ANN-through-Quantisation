# ANN-through-Quantisation
Approximate Nearest Neighbour search through product quantisation of the vectors.

Clone the repo:
```
git clone https://github.com/C-Ritam98/ANN-through-Quantisation.git
```
Install dependencies:
```
cd ANN-through-Quantisation
pip3 install -r requirements.txt
```
Number of clusters = 256 [config.py]

Number of subspaces = 10 [config.py]

Run the server.py file:
```
python3 server.py
```

Output:
```
Dimension of each vector: 100
Number of vectors : 71290
The size consumed by the vectors : 28516128
[INFO] Computing K-means clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] Computing K-means sub-clusters !!
[INFO] K-means clusters computation done !!
The size consumed by the quantised vectors : 128
```
Clearly, after quantisation the size reduces from 28516128 Bytes to 128 Bytes, which is 222782x reduction in size.
