# What does BERT Learn from Multiple-Choice Reading Comprehension Datasets?

2019.12.13

## 主要思想

利用不同的对数据施加扰动的方式，从两个方面评估BERT fine-tune模型：

1. 扰动对人类来说是不影响或影响很小的，这时模型的性能是否会下降？
2. 扰动对人类影响很大时（从文本中已经学不到有效信息），这时模型性能对比人类如何？

如果结果和人类相差较大，说明BERT对文本信息的处理模式不同于人类行为，或者说BERT并不能真正地理解文本信息。

## Unreadable Data Attack

目标：利用黑盒探测方式探究BERT模型依赖于文本数据哪方面的特征。

在测试过程中添加带有**关键词**的干扰信息。如果模型性能显著下降，可以推断模型严重依赖于关键词匹配。干扰信息主要是不可读句子的形式（通过打乱原始输入的单词顺序得到）。

"AddSent"方法：干扰的句子直接加到输入序列的尾部

ShuffleDegree：干扰句子的最小打乱程度，定义为最小编辑距离占序列长度的比例，经验值为0.65

三种干扰方式：AddSent2Pas-Shuffle 、AddSent2Opt-Shuffle 、AddAns2Opt-Shuffle ；即加句子到passage中、加句子到选项中、加答案到选项中。

- AddSent2Pas-Shuffle ：**问题+错误选项**拼成的句子打乱后加到文章里
- AddSent2Opt-Shuffle ：**文章随机选一个句子+错误选项**打乱后生成新的options
- AddAns2Opt-Shuffle ：**错误选项+正确答案** 打乱后作为新的options

实验结果显示，BERT容易受到不可读信息的干扰，这表明BERT十分依赖于关键词匹配这类统计模式。

## Unanswerable Data Training

目标：利用减少输入信息和打乱输入序列的方式构造不可回答的样本（对人类来说），探究对模型来说哪部分信息是需要的。

首先采用部分训练的方式——删除输入中的文章或是问题或是二者都删除（为什么可以都删除？还是说保留了一些passage的文本？）；第二个方式是，采用完全打乱的输入来训练BERT。第一种叫Partial Data；第二种叫Shuffled Data。

### Shuffled Data

打乱原始文本中的单词顺序可以破坏正确的语法结构，从而使问题变得不可回答。但是这种情况下原始的关键词其实仍保留在文本中。采用三种方式打乱：

- P-Shuffle：打乱文章
- Q-Shuffle：打乱问题
- PQ-Shuffle：打乱文章和问题

### Partial Data

- P-Remove：移除文章
- Q-Remove：移除问题
- PQ-Remove：删除二者

可以看到这两种设置下人类都无法从训练样本中学到有用的信息了。然而，在两种训练方式下BERT的效果仍然比random guess好很多，甚至接近原始训练方法下的模型性能。这表明BERT不需要正确的句法信息来回答问题，并且说明数据集中存在统计提示，因此即使没有足够的上下文，BERT也可以表现良好。

## 结论

1. BERT用于选择题型阅读理解任务时，十分依赖于某些关键词。
2. BERT不需要正确的句法和语义信息也可以在该任务上取得较好的性能。
3. BERT在partial training上仍保持较好效果，这表明BERT可以利用数据集中的artifacts漏洞和统计线索来选择正确答案，而并没有真正学习到自然语言理解和推理的过程。

