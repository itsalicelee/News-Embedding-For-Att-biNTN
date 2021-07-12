# Generating News Embedding using Bloomberg News for Att-biNTN
This is an implementation of generating the embedding in Att-biNTN.

For more details, please refer to the reference.
## Links

[GitHub Link](https://github.com/itsalicelee/News_Embedding_For_Att-biNTN)

[Colab Link](https://colab.research.google.com/drive/1Zsy4ppI2HUzj8v5fZJMjr42KaUnNK0-9?usp=sharing)



![Imgur](https://i.imgur.com/12oa9WF.png)

![Imgur](https://i.imgur.com/XO053aX.png)

## Description


|--README (this file) <br />
|--readfile.py <br /> 
|--tag.py <br />
|--tag2Vec.py <br />
|--model.py <br />
|--results <br />
&emsp;|--tdt.pkl (generated from readfile.py) <br />
&emsp;|--tagging.pkl (generated from tag.py) <br />
&emsp;|--ttv.pkl  (generated from tag2Vec.py) <br />
|--20061020_20131126_bloomberg_news <br />
&emsp;|--2006-10-20 <br />
&emsp;&emsp;|--news1 <br />
&emsp;&emsp;|--news2 <br />
&emsp;&emsp;|--... <br /> 

## Methodology

- First of all, we find out the financial news which match the tickers

- Then, we capture the title of these news, remove symbols, and use **Stanford OpenIE** to separate a sentence in to three parts: subject, relation, object

- Utilizing Word2Vec model from genism, we choose **glove-wiki-gigaword-100** as pretrained model, and change each word into a 100-dim vector

- *S1* is computed by the following method, where *e1* is the subject, *r* is the relation, *T1* is a bilinear tensor, *f* represents *tanh*, and *b* is the bias vector

    <img src="https://render.githubusercontent.com/render/math?math=S_1 = f(e_1^TT_1^{[1:k]}r %2B W\begin{bmatrix} e_1 \\r  \end{bmatrix} %2B b)">
    
    Also, 
    <img src="https://render.githubusercontent.com/render/math?math=g(E_1, R) = g(S_1) = U^TS_1">


- *S2, S3, S4, C, Cinv* are computed similarly

- For training goal, we randomly replace either Subject or Object with any word of all the titles in the data and generate the loss function as below:
  
    <img src="https://render.githubusercontent.com/render/math?math=L=max(0, 1-G(E)+G(E^r )) %2B \lambda||\phi||_2^2">
    where 
    <img src="https://render.githubusercontent.com/render/math?math=G(E)=g(C) %2B g(C_{inv} )">

## Usage

- First, download the [Bloomberg news data](https://github.com/philipperemy/financial-news-dataset) and unzip in the folder.
- Run the following command in order:
  - ``python readfile.py``
  - ```python tag.py```
  - ```python tag2Vec.py```
  - ```python model.py ```

- Finally, using results/record, we can further make prediction using the model proposed in the work
- NOTICE: We can use the command ```python read.py [file.pkl]``` to print out the files in ./results

## Reference

D. Daiya, M. -S. Wu and C. Lin, "Stock Movement Prediction That Integrates Heterogeneous Data Sources Using Dilated Causal Convolution Networks with Attention," *ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 8359-8363, doi: 10.1109/ICASSP40776.2020.9053479.
