# Cifar10_Classification
## 程式簡介
### 簡述
* 使用「Convolutional Neural Network, CNN」實作 Cifar10 資料集的「分類」問題

* 正確率約略：0.78 ，如要更高可以參考：https://zhuanlan.zhihu.com/p/49180361
> 此篇適合 CNN 與 人工智慧基本技術 的初學者


### 範例圖
* **Model Structure**    
  ![image](https://user-images.githubusercontent.com/86537930/132137447-9c28c5ce-9bf2-4476-88c2-0c6f362cd7c7.png)


* **Final Result**  
  ![image](https://user-images.githubusercontent.com/86537930/132137439-ce56bf2b-2fa6-4703-85cf-bbe62583b28b.png)

## 資料類別編碼
> 在訓練AI模型前，會針對「類別」的資料做前處理，主要為兩種方法( 還有很多其他方法，有機會再介紹 )
* Label Encoding：把每個類別 mapping 到某個整數，不會增加新欄位  

* One Hot Encoding : 為每個類別新增一個欄位，用 0/1 表示是否  
### Label Encoding
> 例如：下圖中的country欄位，三個國家都被數字0、1、2取代。

![image](https://user-images.githubusercontent.com/86537930/132143944-2295ee1a-de6d-4d98-a161-d92d35ef9987.png)  
	:arrow_down:  
![image](https://user-images.githubusercontent.com/86537930/132143949-a6aa6adc-008f-42d9-a805-573826c99ac7.png)

* 類似於流水號，依序將新出現的類別依序編上新代碼，已出現的類別編上已使用的代碼

* 能確實轉成分數，但缺點是分數的大小順序「常常」是沒有意義的
  * 上述：country本質是類別，並沒有順序大小之分，這樣做模型會認為country之間存在著0 < 1 < 2的關係
  
* 可以使用 sklearn 的套件：
	```python
	from sklearn.preprocessing import LabelEncoder
	labelencoder = LabelEncoder()
	data_le = pd.DataFrame("(Own DataFrame)")  
	data_le["(LabelEncode column)"] = labelencoder.fit_transform(data_le["(LabelEncode column)"])
	```
### One Hot Encoding
> 例如：下圖中的country欄位，獨立出同等類別數量的欄位，並用0/1 表示是否
 
![image](https://user-images.githubusercontent.com/86537930/132143944-2295ee1a-de6d-4d98-a161-d92d35ef9987.png)  
	:arrow_down:  
![image](https://user-images.githubusercontent.com/86537930/132144765-463dd275-84b2-4572-996f-9cfde5319b15.png)

* 為了改良數字大小沒有意義的問題，將不同的類別分別獨立為一欄
* 缺點是需要較大的記憶空間與計算時間，且「類別欄位過多」或「單一類別過多」時越嚴重

### 結論
> 兩種編碼的使用時機，可以依照下面兩種情況選擇
* 原始資料是有序離散值的話 => Label Encoding；原始資料是無序離散值的話 => One Hot Encoding

* 預設採用Label Encoding，除非該特徵重要性高，且空間與時間負擔較低時，才應考慮使用One Hot Encoding

### 補充
如果要換成 mnist 的資料集( 正確率約略：0.98 )，更改下列兩行即可：
```python
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() 

# x_train、x_test維度為(ID, width, height)
```

同時下述程式碼視情況加入：
* CNN Backend 的默認圖像維度順序可能為以下兩種：
	* ‘channels_last’ ：(ID, width, height, channel)
	* ‘channels_first’：(ID, channel, width, height)

* 因此必須將x_train 及 x_test (ID, width, height)增加channels維度，由原本三維轉為四維以符合CNN的需求

* RGB圖片的格式為width, height, channels，MNIST圖片為灰階因此其channel為1

```python
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```


