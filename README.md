# dnn-im2col

--- 

## これは何？

[cuDNNで採用されているIm2Colの手法](https://arxiv.org/pdf/1410.0759.pdf) を実験するために書いたC++で書いたミニマルなDNNフレームワークです。
畳み込みを密行列の積（BLASのGEMM）に変換して高速化しているのが特徴です。
BLASをOpenBLASやcuBLAS、pzBLASなどに置き換えると高速化できます。
