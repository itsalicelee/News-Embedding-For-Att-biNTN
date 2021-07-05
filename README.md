# Generating News Embedding using Bloomberg News for Att-biNTN

This is a work of generating the embedding in Att-biNTN.

For more details, please refer to the reference.

![Screen Shot 2021-07-05 at 7.23.01 PM](/Users/alicelee/Library/Application Support/typora-user-images/Screen Shot 2021-07-05 at 7.23.01 PM.png)

![image-20210705194202995](/Users/alicelee/Library/Application Support/typora-user-images/image-20210705194202995.png)

## Description

|-- embedding
    |-- README (this file)
    |-- readfile.py
    |-- tag.py
    |-- tag2Vec.py
    |-- model.py
    |-- results
        |-- tdt.pkl (generated from readfile.py) 
        |-- tagging.pkl (generated from tag.py)
        |-- ttv.pkl  (generated from tag2Vec.py)
    |-- 20061020_20131126_bloomberg_news
        |-- 2006-10-20
            |-- news1
            |-- news2
            |-- ...

## Methodology

- First of all, we find out the financial news which match the tickers

- Then, we capture the title of these news, remove symbols, and use **Stanford OpenIE** to separate a sentence in to three parts: subject, relation, object

- Utilizing Word2Vec model from genism, we choose **glove-wiki-gigaword-100** as pretrained model, and change each word into a 100-dim vector

- *S1* is computed by the following method, where *e1* is the subject, *r* is the relation, *T1* is a bilinear tensor, *f* represents *tanh*, and *b* is the bias vector

  $S_1 = f(e_1^TT_1^{[1:k]}r + W\begin{bmatrix} e_1 \\r  \end{bmatrix}+ b )$

  $g(E_1, R) = g(S_1) = U^TS_1$

- S2, S3, S4, C, C_inv are computed similarly

- For training goal, we randomly replace either Subject or Object with any word of all the titles in the data and generate the loss function as below:

  $L=max(0, 1-G(E)+G(E^r ))+λ||Φ||_2^2$

  $where\ G(E)=g(C)+g(C_{inv} )$

## Usage

- First, download the [Bloomberg news data](https://github.com/philipperemy/financial-news-dataset) and unzip in the folder.
- Run the following command in order:
  - ``python readfile.py``
  - ```python tag.py```
  - ```python tag2Vec.py```
  - ```python model.py ```

- Finally, using results/record, we can further make prediction using the model proposed in the work

## Reference

D. Daiya, M. -S. Wu and C. Lin, "Stock Movement Prediction That Integrates Heterogeneous Data Sources Using Dilated Causal Convolution Networks with Attention," *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 8359-8363, doi: 10.1109/ICASSP40776.2020.9053479.
