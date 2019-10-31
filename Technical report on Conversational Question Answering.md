[Technical report on Conversational Question Answering]( https://arxiv.org/pdf/1909.10772.pdf )  

本文提出一个RoBERTa+AT+KD的新框架用于对话问答任务，主要涉及的技术有依据标注(rationale tagging)多任务学习、对抗训练、知识蒸馏以及一个语言学的后处理策略。

## 1. Contributions

1. 提出一个CQA任务通用的微调预训练模型：（1）利用答案的依据片段（rationale）中的有效信息进行依据标注（rationale tagging）多任务学习；（2）通过对抗训练加强模型面对扰动的鲁棒性；（3）利用知识蒸馏技术可以充分利用训练良好的模型的额外训练信息。

2. 分析了包括我们的系统在内的抽取式模型的局限性。为了弄清抽取式模型的提升空间，我们在一些自由形式的问答数据集上评估了抽取式模型的性能上界。

3. 我们的系统在不使用数据增强的情况下，达到了SOTA的结果。

## 2. 相关工作

### MRC

现有的问答数据集可以根据每个问题是否与对话历史有关分为单轮问答和多轮问答。许多MRC模型一开始提出都是用于处理单轮问答任务，如BiDAF（Seo等人, 2016），DrQA（Chen等人, 2017）, R-Net（Wang等人, 2017）和 QANet（Yu等人, 2018）。多轮问答的模型有FlowQA（Huang等人, 2018）和 SDNet（Zhu等人, 2018）。

### 预训练模型

预训练语言模型在许多NLP任务中取得了优异的结果，包括GPT（Alec Radford and Sutskever., 2018）, BERT（Devlin等人, 2018）, XLNET（Yang等人, 2019） 和 **RoBERTa**（Liu等人, 2019）等模型。其中RoBERTa的性能可以达到甚至超过所有post-BERT的方法。

## 3. 方法

### 3.1 RoBERTa（Robustly optimized BERT pretraining Approach）

不同于其他问答数据集，CoQA数据集是对话形式的问答。每个问题与之前对话历史相关，因此对于第k轮的问题 $Q_k $，可以重新定义为以下形式：
$$
Q^*_k=\{Q_1,A_1,...,Q_k-1,A_k-1,Q_k\}
$$

并且在每个问题和答案前分别加入 $[Q]$ 和 $[A]$ 标记。给定文本 $C$，RoBERTa的输入就是$[CLS]Q^*_k[SEP]C[SEP]$。

CoQA数据集的答案可以是自由形式、YES、NO或Unknown，除了Unknown答案外，每个答案都有对应的依据片段（rationale）。考虑到数据集的特点，我们在RoBERTa之上采用了一个带YES/NO/Unknown分类器的抽取式方法构建了输出层。首先在训练过程中，选择rationale中具有最高F1得分的文本片段标记为gold answer。在baseline模型中，我们采用一个全连接层得到起始位置logits $l^s$ 和终止位置logits $l^e$ ;对于Yes/No/Unk分类，也简单地在RoBERTa的pooled output $h^p$ 之上添加一个全连接层得到三个logits $l^y$，$l^n$，$l^u$。目标函数定义如下：
$$
p^s=softmax([l^s,l^y,l^n,l^u])
$$

$$
p^e=softmax([l^e,l^y,l^n,l^u])
$$

$$
L_{Base}=-\frac{1}{2N}\sum_{i=1}^{N}{(log p_{y_i^s}^s+log p_{y_i^e}^e)}
$$

### 3.2 RationaleTagging Multi Task

为了利用rationale中的信息，我们加入了一个依据标注任务，预测paragraph中的每个token是否包含于rationale片段中。换句话说，rationale中的token标记为1，其他的标记为0。对于无法回答问题，全部标记为0（无法回答问题没有rationale）。

因此，除了预测答案边界和分类任务，我们另外加入了一个全连接层计算rationale标记的概率。
$$
p_t^r=sigmoid(w_2Relu(W_1h_t))
$$
这里 $h_t\in\mathbb{R}^d$ 是RoBERTa的输出中对应第 $t$ 个token的向量表示。$W_1\in\mathbb{R}^{d\times d}$，$w_2\in\mathbb{R}^d$。ratinale损失定义为如下的平均交叉熵。
$$
L_{RT_i}=-\frac{1}{T}\sum_{t=1}^{T}(y_{it}^rlogp_{it}^r+(1-y_{it})log(1-p_{it}^r))
$$

$$
L_{RT}=\frac{1}{N}\sum_{i=1}^NL_{RT_i}
$$

通过联合最小化两个目标函数训练模型：
$$
L=L_{Base}+\beta_1L_{RT}
$$
此外，rationales可以用于辅助Yes/No/Unk问题的分类，过程如下：首先将 $p_t^r$ 和 $h_t $ 相乘得到rationale表示 $h_t^r$，然后使用一个注意力层得到附加了注意力的rationale表示 $h^{p*}$。
$$
h_t^r=p_t^r\times h_t
$$

$$
a_t=softmax_{t=1}^T(w_{a2}Relu(W_{a1}h_t^r))
$$

$$
h^{p*}=\sum_{t=1}^Ta_t\times h_t
$$

这里 $W_{a1}\in\mathbb{R}^{d\times d}$， $w_{a2}\in\mathbb{R}^{d}$ 是可学习参数。将baseline中分类器的输入由 $h^p$ 替换为 $[h^{p*};h^p]$，即$h^{p*}$ 和 $h^p$  的拼接，最后经过全连接层产生 $l^y$，$l^n$，$l^u$。这里 $h^p$ 即上述提到的RoBERTa的pooled output。**这里的思想可以简单理解为用注意力机制将rationale标记任务得到的概率分布引入三分类任务，以辅助三分类的预测。**

### 3.3 对抗训练和虚拟对抗训练

对抗训练（Goodfellow等人, 2014）是一种用于神经网络正则化的方法，可以帮助模型对抗噪声，从而提高泛化能力。

- #### Adversarial Training (AT) 

我们通过对嵌入层施加细微的扰动来生成对抗样本。假设 $v_w$ 是单词 $w$ 的嵌入向量，$\widehat{\theta}$ 表示模型当前参数，则对抗嵌入向量为 $v_w^*$（Miyato等人，2016）：
$$
g_w=-\bigtriangledown_{v_w}L(y_i|v_w;\widehat{\theta})
$$

$$
v_w^*=v_w+\epsilon g_w/\|g_w\|_2
$$

先由损失函数计算各个词嵌入向量的梯度，再根据梯度对词嵌入向量进行更新。这里 $y_i$ 是gold label，$\epsilon$ 是控制扰动大小的超参数。接着，对抗损失可以计算如下：
$$
L_{AT}(\theta)=-\frac{1}{N}\sum_{i=1}^NCrossEntropy(\cdot |V^*;\theta)
$$
$V^{*}=[v_{w1}^{*},...,v_{wn}^{*}]$ 是对抗嵌入矩阵。

- #### Virtual Adversarial Training (VAT)  

虚拟对抗训练和对抗训练类似，但采用了**非监督的对抗扰动**。为了获取虚拟对抗扰动，我们先对单词的嵌入表示加入一个高斯噪声：
$$
v_w^{'}=v_w+\xi d_w
$$
$\xi$ 是一个超参数，$d_w\in\mathbb{R}^d$ 服从正态分布（$d_w \sim\mathcal{N}(0,I)$）。那么 $p(V)$ 和 $p(V^{'})$ 的KL松散度函数[^3]的梯度可以按下面的公式估计：
$$
g_w=\bigtriangledown_{v^{'}}D_{KL}(p(\cdot|v_w;\widehat{\theta})\|p(\cdot|v_w^{'};\widehat{\theta}))
$$
接着，与对抗训练相似，对抗扰动被加入词嵌入向量：
$$
v_w^*=v_w+\epsilon g_w/\|g_w\|_2
$$
最后，虚拟对抗损失计算如下：
$$
L_{VAT}(\theta)=\frac{1}{N}\sum_{i=1}^{N}D_{KL}(p(\cdot |V;\theta)\|p(\cdot |V^*;\theta))
$$
$V^*$ 是对抗嵌入矩阵。

注意到对抗训练和虚拟对抗训练用了不同的评估函数，即**交叉熵函数** $CrossEntropy()$ 和**KL松散度（相对熵）函数** $D_{KL}()$。这是因为前者是有监督的学习，可以直接用交叉熵计算预测值和真实标签之间的差距；后者是非监督学习，只能通过评估加入扰动前后的预测概率分布的差异程度来衡量预测值和实际值的差距。

[^3]: KL松散度又称相对熵，用于衡量两个概率分布之间的不同程度，与交叉熵类似。

- #### 损失函数

整体损失通过简单将各部分损失相加得到：
$$
L=L_{Base}+\beta_1L_{RT}+\beta_2L_{AT}+\beta_3L_{VAT}
$$

### 3.4 知识蒸馏

通过将teacher model的输出作为student model训练目标，知识蒸馏技术（Furlanello等人，2018）能够将“知识”从一个机器学习模型（teacher）迁移到另一个模型（student）上。

#### Teacher Model  

本文采用的teacher model就是前述方法。

#### Student Model 

利用teacher model的输出概率 $f(x^i,\theta^\tau) $ 作为额外监督标签对student model进行训练。我们通过不同的随机种子设置训练了几个teacher model，teacher label计算如下：
$$
p_i^{kd}=\frac{1}{T}\sum_{\tau=1}^{T}f(x^i,\theta^\tau)
$$
$\theta^\tau$ 是第 $\tau$ 个teacher model的参数，$T$ 是teacher model的总数。KD loss定义为 $p_i^{kd}$ 和 $f(x^i,\theta^s)$ 的交叉熵：
$$
L_{KD}(\theta^s)=-\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^Tp_{it}^{kd}logf(x_{it},\theta^s)
$$
$\theta^s$ 是student model的参数。因此，student model的总损失定义如下：
$$
L_S=L_{Base}+\beta_1L_{RT}+\beta_2L_{AT}+\beta_3L_{VAT}+\beta_4L_{KD}
$$

### 3.5 后处理 

由于我们的模型是抽取式的，所以无法处理多选类型的问题。对于多选类问题，模型可能抽取到和选项同样的单词，只是形式有所不同，例如：选项是'walk'和'ride'，而模型抽取到的span是‘walked’。

基于词语相似度的后处理过程可以用于缓解该问题。首先通过语言学规则从问题中抽取合理的选项；接着计算每个选项的词嵌入和答案tokens的余弦相似度；最后选择具有最大相似度的选项作为答案。
$$
ans=\mathop{arg\,max}_o\{sim(o,a)|o\in\mathbb{O},a\in\mathbb{A}\}
$$
$\mathbb{O}$ 和 $\mathbb{A}$ 分别是选项和答案的词嵌入集合，$sim(o,a)$ 表示他们的余弦相似度。

## 主要参考文献

### RoBERTa

Liu Y, Ott M, Goyal N, et al. [Roberta: A robustly optimized bert pretraining approach]( https://arxiv.org/pdf/1907.11692.pdf)[J]. arXiv preprint arXiv:1907.11692, 2019. 

### 对抗训练

Goodfellow I J, Shlens J, Szegedy C. [Explaining and harnessing adversarial examples]( https://arxiv.org/pdf/1412.6572.pdf)[J]. arXiv preprint arXiv:1412.6572, 2014. 

Miyato T, Dai A M, Goodfellow I. [Adversarial training methods for semi-supervised text classification]( https://arxiv.org/pdf/1605.07725.pdf)[J]. arXiv preprint arXiv:1605.07725, 2016. 

### 知识蒸馏

Furlanello T, Lipton Z C, Tschannen M, et al. [Born again neural networks]( https://arxiv.org/pdf/1805.04770.pdf)[J]. arXiv preprint arXiv:1805.04770, 2018. 
