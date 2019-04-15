**BERT预训练模型论文《[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)》的阅读笔记**

[TOC]

## 一、Model Architecture 

### 规模

- BASE：L=12,H=768,A=12,参数量=1.1亿
- LARGE：L=24,H=1024,A=16,参数量=3.4亿

### BERT、GPT、ELMo三者的区别

![](https://img2018.cnblogs.com/blog/1018727/201903/1018727-20190310211134355-849402575.jpg)

| 模型                        | 结构                                      |
| --------------------------- | ----------------------------------------- |
| BERT                        | 双向的Transformer（Vaswani et al., 2017） |
| GPT（Radford et al., 2018） | 从左到右的Transformer                     |
| ELMo（Peters et al., 2018） | 浅层连接、独立训练的两个方向LSTM          |

## 二、Input Representation 

![](https://img2018.cnblogs.com/blog/1018727/201903/1018727-20190310211233454-978103877.jpg)

**单词，段，位置（token, segment, position）** 三者的嵌入表示，共同构成了输入表示。

1. **token embeddings**采用一个包含30,000个token的词表对单词进行嵌入表示，如图需要特别指出的是这里采用的是WordPiece embeddings（Wu et al., 2016），切割后的单词碎片带有`##`。
2. 采用一个已学习的位置嵌入表示（**position embeddings**），支持512个token的序列长度。
3. 输入序列的起始标记是特殊的分类embedding，`[CLS]` ，用于表征整个句子。对于分类任务，对应于该标记的最终隐藏状态（即Transformer的输出）被作为聚集序列表示（aggregate sequence representation ）。
4. 一个输入序列包含多个句子，通过两种方式区分不同句子：句子间添加特殊标记，`[SEP]` ；对每个token添加一个已学习的**segment embedding**：若属于句子A，则标记为A；若属于句子B，则标记为B。对于单句输入，只用句子A的embedding即可。

## 三、预训练任务 

预训练任务由两个子任务组成。

### 任务1 Masked LM 

随机掩盖15%的WordPiece tokens，用`[MASK]`标记替代，然后对这些被掩盖的tokens进行预测。该任务可以看作一个完形填空任务。

该方法虽然可以构建一个双向预训练模型，但是存在两个缺点：一是预训练和微调（fine-tuning ）之间存在不一致问题，原因是如果对选中的单词全部进行mask，`[MASK]`标记在下游NLP任务中从未出现，这会引导模型认为输出是针对这个特殊标记，导致作出错误的预测；二是每个batch只预测15%的单词，这样的训练过程需要极大的训练步才能收敛。

为了解决第一个问题，采取下述策略。随机选中token后，Mask过程由数据生成器控制，过程如下：

- 80%的情况下，将单词替换为`[MASK]`标记，如：

```
my dog is hairy --> my dog is [MASK]
```

- 10%的情况下，将选中的单词随机替换成另一个词，如：

```
my dog is hairy --> my dog is apple
```

这部分在所有token中只占了1.5%，不至于使语言模型的性能受到太大损害。

- 10%的情况下，不做替换，如：

```
my dog is hairy --> my dog is hairy
```

这样做的目的是使表达偏向于实际观察到的单词。

### 任务2 Next Sentence Prediction 

对于某些下游任务，理解句子间的关系至关重要，因此提出此任务。

对每个预训练样本，构造一个句子对A和B，并标记句子间的关联。其中50%的B是A实际的下一句，50%的B是从语料库随机抽取的。如下：

```
# example1
Input = [CLS] the man went to [MASK] store [SEP]
		he bought a gallon [MASK] milk [SEP]
Label = IsNext

# example2
Input = [CLS] the man [MASK] to the store [SEP]
		penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

## 四、预训练过程

预训练过程和现有预训练语言模型方法基本相同。预训练语料采用BooksCorpus (800M个单词) 和Wikipedia (2,500M个单词）。

基本过程包含：

1. 采样：每个样本有A,B两个句子，如上述样例；
2. WordPiece分词；
3. LM masking：遮掩率统一为15%；
4. 训练：对两个子任务的目标函数进行联合学习。

## 五、Fine-tuning 过程

在预训练模型的基础上，只需额外添加一层输出层，就可应用到指定的下游任务。如下所示分别是对四种不同下游NLP任务的fine-tuning。

- (a) 句子关系推断任务
- (b) 单句分类任务
- (c) 问答任务
- (d) 序列标注任务

![](https://img2018.cnblogs.com/blog/1018727/201903/1018727-20190310211336373-1466765788.jpg)

`Our task specific models are formed by incorporating BERT with one additional output layer, so a minimal number of parameters need to be learned from scratch. Among the tasks, (a) and (b) are sequence-level tasks while (c) and (d) are token-level tasks. In the figure, E represents the input embedding, Ti represents the contextual representation of token i, [CLS] is the special symbol for classification output, and [SEP] is the special symbol to separate non-consecutive token sequences.` 

## 六、BERT和OpenAI GPT的比较

|                 | BERT                                                        | GPT                                                     |
| --------------- | ----------------------------------------------------------- | ------------------------------------------------------- |
| **语料**        | BooksCorpus & Wikipedia                                     | BooksCorpus                                             |
| **token**       | 在预训练时就对[SEP], [CLS]和句子A/B标记 embeddings进行学习  | 仅在fine-tuning阶段引入句子分隔符[SEP],分类器token[CLS] |
| **训练规模**    | 1M训练步，每个batch包含128,000个单词                        | 1M训练步，每个batch包含32,000个单词                     |
| **fine-tuning** | 对指定任务挑选适合的fine-tuning学习率，在开发集上有更佳表现 | 学习率都是5e-5                                          |

## 主要参考文献

1. *Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. **Attention is all you need.** In Advances in Neural Information Processing Systems, pages 6000–6010.* 
2. *Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. **Improving language understanding with unsupervised learning.** Technical report, OpenAI.* 
3. *Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. **Deep contextualized word representations.** In NAACL.* 
4. *Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. **Google’s neural machine translation system: Bridging the gap between human and machine translation.** arXiv preprint arXiv:1609.08144.* 
5. [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)