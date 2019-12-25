2019.12.22

整理一下对抗训练相关的几篇论文

------

## Robust Machine Comprehension Models via Adversarial Training（NAACL2018）

### 摘要

现有MRC模型缺乏鲁棒性，在基于AddSent算法的对抗攻击下，F1得分下降了50%（以SQuAD为例）。**而在AddSent生成的数据上重新训练模型，对泛化性能影响有限**（AddSent的作者尝试在AddSent生成的对抗样本上重新训练BiDAF模型，但并不是很有效）。本文提出了AddSentDiverse，一种创新的对抗样本生成方法，通过提供有效的对抗样本来惩罚模型作出某些浅层假设的行为，从而显著增加对抗训练数据间的方差。进一步地，我们提升了模型的语义关系学习能力，以增强对AddSent的语义扰动（如反义词）的鲁棒性。

### AddSent

1. 生成语义修改的问题：基于反义词或命名实体的语义置换
2. 通过一个**固定的 fake answers set** 生成假答案，类型和原答案一样（如：北京-->上海）
3. 经过人工制定的规则将修改后的问题和假答案结合到一起
4. 人工修正语法错误，得到最后的干扰句（distractor）
5. 将干扰句加到文章的末尾

这种方式生成的对抗样本特点太明显，用于训练时模型很容易利用其表面的统计线索，直接忽视添加的干扰信息，从而使对抗训练失效（干扰信息是最后一句，并且假答案来自一个固定的集合，模型极易捕捉到这种模式）。

### AddSentDiverse

本文提出的对AddSent的改进方法。改进点有两个：Random Distractor Placement  和 Dynamic Fake Answer Generation 。

#### Random Distractor Placement 

干扰句子加入的位置随机，这样可以保证不引入**句子的位置与是否包含答案正确答案存在相关性**这种信息。

#### Dynamic Fake Answer Generation

将训练集中所有答案组成的集合定义为S，并事先标记每个答案的类型（人物，地点等），每次从S中随机抽取一个同类型的答案作为假答案。这样扩大了答案的选取范围，从而降低了直接通过某些关键词直接判别文本是否是干扰信息的可能性。

### 语义特征增强模型

考虑到模型对干扰信息和原始问题的语义区别学习能力有限，利用语义特征进一步增强对抗训练模型对干扰信息的鉴别能力。（AddSent是基于语义的对抗攻击，如反义词风格的干扰信息）

具体方法是利用**WordNet**在输入序列的每个单词上添加**两个特征**，指示在文章/问题中是否存在**同义词和反义词**。从而使得模型可以直接利用词汇级的语义信息，而非通过学习词嵌入的统计关系获取这种信息。

### 实验

在SQuAD和四种对抗数据集上进行了实验：AddSent（distractor 加在末尾），AddSentPrepend（distractor 加在前面）,  AddSentRandom（distractor 加入位置随机） ，AddSentMod （采用了一个不同的fake answers set ）。 

模型选用BiDAF + Self-Attn + ELMo (BSAE)  ，分别用原始的SQuAD数据，加入了AddSent对抗样本的SQuAD数据，加入了AddSentDiverse对抗样本的SQuAD数据，三种训练集进行训练。只有最后一种模型在各个对抗数据集上都有较好性能，表现出整体的鲁棒性提升。

为了评估不同的distractor添加策略带来的影响，BSAE模型分别在四种不同的distractor添加策略下增强的数据集上进行对比试验：distractor加在前面、中间、末尾和随机位置。结果表明，Random方式在AddSent和AddSentPrepend测试集上均表现较好。

为了评估用动态生成fake answer的策略取代从事先定义的集合中选择fake answer的效果，两种方法训练的模型分别在AddSentPrepend，AddSentMod上进行实验，发现前者在两个测试集上效果更好且基本不下降，而后者在第二个测试集上性能drop了3%。

加入了基于WordNet的语义关系指示特征后，模型在原始数据集上性能基本不下降了（和用原始数据集训练的模型相比），并且在原始数据集和对抗数据集上均有提升。

------

## Robust Neural Machine Translation with Doubly Adversarial Inputs（ACL2019）

这篇和 Miyato 的《Adversarial Training Methods for SemiSupervised Text Classification》很像，都是基于损失梯度找到有效的对抗样本，只不过后者是对词嵌入直接加扰动生成对抗样本，本文则是从词表中搜索一个合适的替代词。

### 摘要

本文提出一个提升NMT鲁棒性的方法，包括两部分工作：(1) 用 adversarial source examples  攻击翻译模型；(2) 利用 adversarial target inputs  强化翻译模型，提高模型对adversarial source inputs 的鲁棒性。提出一种基于梯度的对抗样本生成方法，由clean inputs得到的翻译损失计算梯度（这和 Miyato 的方法有什么区别？）。

NMT模型一般是seq2seq的结构，所以可以拆分为编码器和解码器两部分。编码器将源语言输入序列 X 映射为一个隐藏状态表示序列 H。解码器将目标语言输入序列 Z 转化为输出序列 Y，即最终的翻译序列。Z 是 Y 已经产生的部分，即用已经解码的序列作为下一步解码的输入。总的来说NMT有两部分输入，X 为 Source Inputs，Z 为Target Inputs 。

###  Attack with Adversarial Source Inputs  

基本思想是对编码器的输入序列 X 生成一个对抗输入，对NMT模型进行对抗攻击。生成对抗输入的过程是这样的：（1）计算损失项对原始输入中某个词词嵌入的梯度g；（2）从词表中选择一个单词，该单词与原始单词的词嵌入差值为d，d 与 g 的余弦相似度最大。这个思想实际上和 Miyato 挺像的，都是找到沿着损失正梯度方向找到一个干扰性最大的词（词嵌入）。

词表怎么确定呢（不可能用总词表，搜索空间太大）？这里对每个单词定义了一个top_n词表，即最相近的n个单词。怎么评估单词间的相似度呢，这里引入了一个源语言的双向语言模型作为likehood function。

为了保证输出与输入的偏差不能太大，只对输入序列一部分单词进行替换。文中采用的采样方式是均匀分布。

这种产生对抗输入的算法文中称为AdvGen。每个原始样本 (X，Y) 产生一个对抗样本 (X'，Y)，后者一起用于训练。

### Defense with Adversarial Target Inputs  

(X'，Y) 可能会引入误差并累积到解码阶段，对解码预测过程造成比较大的影响，因此对于在解码阶段引入adversarial target inputs 用于抵抗这种误差。同样利用AdvGen算法产生adversarial target inputs，即对 Z 生成 Z‘。

------

## Effective Adversarial Regularization for Neural Machine Translation（ACL2019）

### 摘要

将基于对抗性扰动的正则化方法应用于NMT模型，本文评估了不同的对抗扰动方法的有效性，并表明对抗性扰动技术可以有效提升包括基于LSTM和基于Transformer的NMT模型的性能。

### 主要思想

加扰动的方式就是《Adversarial Training Methods for SemiSupervised Text Classification》中 AT 和 VAT 的方法。应用到机器翻译任务上，就变成了在编码器和解码器的输入词嵌入上都可以加一个扰动。

根据对词嵌入加扰动的方式，有3种configuration：只在编码器输入词嵌入上加扰动；只在解码器输入词嵌入上加扰动；二者都加扰动。产生对抗性扰动的方式不同，有两种configuration：AT 和 VAT。最后的结论是：编码器解码器都加扰动效果最好；VAT效果好于AT。（好像都是意料之中的）

这篇思想比较简单，就是对抗训练+虚拟对抗训练在机器翻译上的一个应用，不过可以借鉴一下作者文章的思路。（主要是Introduction这部分）

先介绍对抗样本和对抗训练的概念——最早在图像处理领域提出和讨论。

NLP中采用对抗训练的困难——文本单词表示是离散的，无法对输入施加细微的扰动。引入Miyato 等人的工作：基于损失的梯度在词嵌入上加扰动的方法，成功地在文本分类上应用了对抗训练并取得效果。**这种方法可以被解释为一种正则化方法，因此本文引入这种正则化技术并称为 adversarial regularization。**

提出本文的目标——在更为复杂的NMT模型中利用这种技术，原因是NMT模型在NLP研究中很重要，广泛应用于其他领域。（这点有点牵强吧，没说到**机器翻译存在什么主要问题需要解决，为什么需要这种技术**）然后作者还说NMT中加扰动的方式有好几种，所以我们的工作不是没有意义的。（。。。fine）

最后总结了下，本文探究了 adversarial regularization 在NMT 模型上的有效性，并希望可以将这个方法作为一种通用技术进一步提高大部分 NMT 模型的性能。本文证明了几种不同 configuration 的对抗正则化方法在两种典型模型上的有效性，LSTM-based 和 Transformer-based 模型。



