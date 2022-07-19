# Depthwise-separable-CNN

## 程式簡介
### 簡述
*  使用 Cifar10 資料集評估 depthwise separable CNN 的「輕量化」效能
![image](https://user-images.githubusercontent.com/93152909/153930286-e7da3891-dfc4-49a0-89fa-d024e51ad2f9.png)

*  使用三種架構下去評估：
	* RegularConv：regular CNN
	* BothConv：regular CNN with part depthwise separable CNN
	* SeparableConv：depthwise separable CNN
	
* 架構部分除了convolution的部分有更改，其他的都一樣，例如：Dropout、MaxPooling2D、BatchNormalization、Dense

* 參數設定，三者一樣：
	* epochs = 150
	* batch_size = 256
	* loss = categorical_crossentropy
	* optimizer = Adam
	* learning rate = 0.001

	
* 正確率約略落在 0.8 ~ 0.9 ，如要更高可以參考：https://zhuanlan.zhihu.com/p/49180361

### 範例圖
* **【Fig1】Validation Loss comparison**    

	表示不同模型在測試集上的Loss，我們其實看不太出來三者的差別，但意義是代表三種模型都會收斂。

![image](https://user-images.githubusercontent.com/93152909/153927425-a8472c35-1d6e-4f9b-861e-ac01bc289300.png)

* **【Fig2】Validation Accuracy comparison**  

	不同模型在測試集上的Accuracy，可以發現RegularConv在表現上依然是最好的(正確率最高)。配合【Fig3】【Fig4】【Fig5】可以發現BothConv與SeparableConv，不管在訓練時間或是訓練參數上都少於RegularConv，幾乎減少一半，但正確率幾乎差距不大。
	> **「因此捨棄些微的模型正確率，換取時間與計算量的大幅減少似乎是投資報酬率很高的一種做法。」**
	
![image](https://user-images.githubusercontent.com/93152909/153927464-56268033-14f0-4639-b2ae-b7a1d4eab946.png)

* **【Fig3】Trainable params ( M ) comparison**  

	表示不同模型的參數量

![image](https://user-images.githubusercontent.com/93152909/153927478-e0c2f0b7-e945-4212-bba4-5d2112fc94e4.png)

* **【Fig4】Train time ( S ) comparison**  

	表示不同模型的訓練時間，很有趣的會發現【Fig4】與【Fig3】有小矛盾，因為照理參數越少，訓練時間應該越短，但實際上，參數最少的SeparableConv，其訓練時間反而長於BothConv。後來經過我的研究推測，因為模型的訓練時間往往會取決於當下電腦的環境，例如有沒有其他資源在同時使用顯卡；亦或是模型的設計是否適合硬體平行加速處理等等，這些因素都會影響模型運算的時間，因此新增【Fig5】來做客觀的比較。

![image](https://user-images.githubusercontent.com/93152909/153927484-d6d6f9fb-1b6e-454a-8ab2-169b6f6ebae6.png)

* **【Fig5】FLOPs ( e+02 G ) comparison**  

	* 表示不同模型的浮點運算數，不因為硬體不同而不同，較客觀
	* FLOPS、FLOPs兩種差別：
		* FLOPS：全大寫，是floating point operations per second的縮寫，指每秒浮點運算次數，是衡量硬體效能的指標
		* FLOPs：s小寫，是floating point operations的縮寫，意指浮點運算數，用於衡量模型的複雜度

![image](https://user-images.githubusercontent.com/93152909/153927501-0f5e21df-55d6-416f-9acf-a340597a1c13.png)

## Depthwise separable convolution
* Depthwise separable convolution的相關概念最早出現在名為「Rigid-motion scattering for image classification」的博士論文中，主要也是用來提取特徵，但相比於常規卷積，其參數量和運算成本較低，主要用於 **「網路結構輕量化」** ，例如：MobileNet 中就有使用其技術來縮小網路結構；Xecption 中也有用，但其主要目的並不是輕量化 。

* Depthwise separable convolution主要由兩部分組成：
	* 「Depthwise convolution」

	* 「Pointwise convolution」
	
###  Regular convolution
原始的CNN設計中，每張Filter會根據輸入圖片的channel產生相同的數量的channel，且每張Filter都會跟輸入圖片做卷積運算，如下圖

![image](https://user-images.githubusercontent.com/93152909/153932191-ce8dd886-9deb-41ce-9287-4ea356cec7c0.png)

###  Depthwise convolution
在Depthwise convolution中，只會產生跟輸入圖片channel相同的Filter數量，同時Filter的channel為1。一張Filter只會對應一個輸入圖片的channel做計算，如下圖。因此相比於Regular convolution，可以少掉3/4的計算量。但Depthwise convolution的缺點就是他無法將圖片擴展到高維度，因為Filter數量只中是根據輸入圖片channel的數量決定，所以通常會搭配Pointwise convolution來使用。

![image](https://user-images.githubusercontent.com/93152909/153932362-74a0edf1-252c-4d18-899c-81a74c05b4d3.png)

### Pointwise convolution
為了改善Depthwise convolution無法擴展到高維度的缺點，所以當初Depthwise convolution的設計者就決定使用1 * 1的Regular convolution來擴展維度，換言之 1 * 1的Regular convolution就是Pointwise convolution，如下圖。

![image](https://user-images.githubusercontent.com/93152909/153932594-6c1095cc-2658-4265-a803-7acbc35bd3ea.png)

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


