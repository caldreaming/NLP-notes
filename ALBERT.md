2019.12.18

# ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

提出两种减少参数的技术可以降低内存占用，提高BERT的训练速度。另外，引入一个自监督损失用于句间连贯性的建模，可以帮助下游任务更好地处理多句输入。ALBERT用比BERT-large更少的参数，在多个benchmark上超过了后者的性能。

扩展现有模型往往会有内存不足、训练速度下降等障碍，现有的解决方法有模型并行、改进的内存管理策略。这可以解决内存不足的问题，但是不能解决通信开销、模型退化等问题。

## A Lite BERT(ALBERT)

### Factorized embedding parameterization  

不论是BERT，还是后来的XLNet、RoBERTa等，WordPiece embedding的大小E总是保持和hidden size H一样，增大H则E也增大。从建模角度来说，WordPiece embeddings 用于学习上下文无关的表示，而隐层的embeddings则是学习上下文相关的表示。针对上下文长度的实验表明，BERT-like表示的强大性能得益于利用上下文学习到的文本表示。因此，本文认为，隐层的embeddings大小应该大于WordPiece embedding的大小以包含更多信息，即H>>E。

NLP任务一般都需要比较大的词典，即词典的size V 很大。如果E随着H一起增加，那么embedding矩阵也会变大(V x E)。这样的话，模型的参数量很快就上去了，但是训练过程中更新的参数却很稀疏。

ALBERT采用**对embedding矩阵因式分解**的方法解决这个问题。对于每个one-hot向量，先经过低维embedding矩阵映射为大小为E的表示，再经过高维的embedding矩阵映射为H维隐层空间的表示（原来是直接从one-hot映射为H维表示）。分解后embedding的参数从**O(V x H)**降低为**O(V x E + E x H)**，当E<<H时参数量可以明显减少。

### Cross-layer parameter sharing  

ALBERT提出一种**跨层的参数共享**方法作为提高**参数效率**的方式。对于Transformer，参数共享有多种方式：仅共享前馈网络(FFN)参数，仅共享注意力相关参数，共享所有参数。ALBERT默认共享所有参数。

比较相邻层输出之间的L2距离和余弦相似度，发现ALBERT每一层输出embedding比BERT的每一层输出embedding变化更小，层与层之间的转化更为平滑。这说明参数共享对稳定网络参数具有一定作用。

和第一点只减少了embedding矩阵的参数相比，第二点降低了模型整体参数量。所以第二点在减少参数量上是更有效的，可以说ALBERT减少参数主要靠的就是第二点方法。

### **Sentence Order Prediction（SOP）**

除了MLM，BERT中设置了另一个任务next-sentence prediction (NSP)  用于预测两段文本是否在原始文本中是连续地出现的，以此学习句子间的连贯性。后续研究发现该任务不能够有效帮助到相关的下游任务，本文认为原因可能是相比于MLM该任务过于简单。NSP实际上包含了”topic prediction”和“coherence prediction”两个目标，但是，与“coherence prediction”相比，”topic prediction”更容易学习，并且在MLM任务在这点上也可以学习到一样的效果。

ALBERT提出SOP任务取代NSP，将NSP中的“coherence prediction”作为主要学习目标，消除了”topic prediction”。SOP的正样本和NSP的获取方式是一样的（同一篇文档中抽取两个连续的段落），负样本把正样本中两个段落的顺序对调即可。这样一来，由于段落都是在同一篇文档中抽取的，就避免了主题相关的影响。

NSP可以利用主题特征做出预测（正样本两个句子来自同一主题，负样本来自不同主题），这样一来“coherence prediction”的目标就没有达到。SOP消除了”topic prediction”的影响，使得模型能够学习到更为准确的句间连贯性特征。

## 参考
[如何看待瘦身成功版BERT——ALBERT？-小莲子的回答-知乎](https://www.zhihu.com/question/347898375/answer/863537122)

[一文揭开ALBERT的神秘面纱](https://blog.csdn.net/u012526436/article/details/101924049)
