---
title: Use Deep learning on tabular data by training Entity Embeddings of Categorical Variables.
date: '2019-10-10'
tags:
  - deep-learning
  - embeddings
  - kaggle
  - pytorch
---

> TLDR; Use Entity embeddings on Categorical features of tabular data from [Entity embeddings paper](https://arxiv.org/pdf/1604.06737.pdf). Code [here](https://github.com/chandureddivari/kaggle/blob/master/Pytorch%20data%20loader%20%26%20category%20embeddings.ipynb)

### Kaggle days

[Kaggle Elo merchant category recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation) being my first competition, my expectations weren't sky high and I'd be very happy if I managed to standout amongst the top 10%. I am trailing at 570 of 4000 odd data scientists in the competition. ~200 more places and I can achieve my target of being amongst top 10% comfortably. I have tried all the ML best practices and tricks known to me. I have done monstrous aggregates of aggregates, bevy of models (hehehe..) like LGBM, XG Boost, Random forests, Catboost and model post processing, parameter tuning, model blending, ensembling, feature permutation, elimination, recursive feature selection, Boruta and many more. I've written about this in detail [here](/posts/elo-merchant-feature/). 

Everyone in the kaggle discussion forums are tight lipped about their secret ingredient for feature engineering. They were already generous about sharing some fantastic ideas like reverse engineering normalized & anonymized data. May be some arcane feature derived from permutation of aggregates of customer's age, merchant value is what I'm missing. I don't know. I've foraged through all popular data science articles, blogs and papers. I've scoured Google for all known *"techniques & tips"*. 

But there's one secret trick up my sleeves that shines like my ray of hope for the jump. It's a little known trick to some kagglers for putting them in top 3 ranks. Some big organisations have already started saving costs. It has not only improved their model performance but also made it more simpler & maintainable. It's everyone's secret weapon against tabular data, but only few talked about it. It's so obvious, so conspicous, so simple yet very effective. Enter deep learning for tabular data. Also it *put me in the top 3% of the competition* (Yay!). 

![Elo merchant competition on kaggle](/images/kaggle-silver.png "Silver in my first kaggle competition")

[Jeremy Howard of fast.ai](http://course17.fast.ai/lessons/lesson4.html) explained this implementation from [paper](https://arxiv.org/pdf/1604.06737.pdf) by Cheng Guo and Felix Berkhahn. Cheng Guo and Felix Berkhahn wrote this paper after winning 3rd place in the Kaggle Rossmann Competition, which involved using time series data from a chain of stores to predict future sales. The 1st and 2nd place winners of this competition used complicated ensembles that relied on specialist knowledge, while the 3rd place entry was a single model with no domain-specific feature engineering.

#### Deep learning to rescue
Deep learning is the new antidote for esoteric feature engineering & domain expertise required for shipping effective models. You need not know about *Lagrange Multipliers* or about *Kullback–Leibler divergence* to get a lead in the competition. Your deep learning model will come up with weights to interpret those for you. Deep learning is creating waves in deep learning & kaggle data science community for long. It is used for winning kaggle competitions. Giants like [Google](https://twimlai.com/twiml-talk-124-systems-software-machine-learning-scale-jeff-dean/) have been using it in production for a while now. [Pinterest](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e) & [Instacart](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc) shared their experiences how they were able to improve their model accuracy by simultaneously reducing the complexity. 

But isn't deep learning particularly effective with semi-supervised learning on unstructured data, you may ask. Yeah but end of the day we are multiplying tensors of calculated weights to make the predictions. Hey, but we do that using multi layer perceptrons you may say. I should be bit more specific, we will use **deep learning on tabular data by training Entity Embeddings of categorical features**. Also multi layer perceptron models don't use non linear activation layers or Batch norm or Dropouts.

### What are embeddings actually?

Embeddings isn't an entirely new concept. We have been using word embeddings for a while now from Word2Vec, Glove, ELMo etc. By using embeddings on categorical features we capture relationships between categories. We will be talking about the findings from this [paper](https://arxiv.org/pdf/1604.06737.pdf) particularly.

#### Applying Embeddings for Categorical Variables
[One hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) has been the go to approach to deal with categorical variables. The problem with one hot encoding is that we would have lot of sparse vectors to handle. An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. 

Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. Perhaps animals like dog and cat share similarities (both are pets), and maybe hovercraft can be tagged along with boats & vehicles. Similarly for zip codes, there may be patterns for zip codes that are geographically near each other, and for zip codes that are of similar socio-economic status.

A way to **capture these multi-dimensional relationships between categories** is to use embeddings. 

![Visualisation of embedding vectors](/images/embeddings-visualisation.png "Visualising embeddding vectors in 3D space")

>Embeddings can also help in approaching clustering problems in a novel way.


This is the same idea as is used with word embeddings, such as Word2Vec. For instance, a 3-dimensional version of a word embedding might look like:

| Name  | vector |
|--------|-----------------|
| dog    | [1.0, 0.2, 0.0] |
| kitten | [0.0, 1.0, 0.9] |
| cat    | [0.0, 0.2, 1.0] |
| puppy  | [0.9, 1.0, 0.0] |


Notice that the first dimension is capturing something related to being a dog, and the second dimension captures youthfulness. This example was made up by [Rachel in her article](https://www.fast.ai/2018/04/29/categorical-embeddings/), but in practice you would use machine learning to find the best representations (while semantic values such as dogginess and youth would be captured, they might not line up with a single dimension so cleanly). 

Similarly we learn embedding vectors for each of the metadata values (quarter hour of the day, day of the week, week of the year, client ID, etc.) and use those embeddings as supplementary inputs to our model.

#### Reusing Pretrained Categorical Embeddings

Embeddings capture richer relationships and complexities than the raw categories. Once you have learned embeddings for a category which you commonly use in your business (e.g. product, store id, or zip code), you can use these pre-trained embeddings for other models. For instance, Pinterest has created [128-dimensional embeddings](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e) (128-dimensional embeddings for its pins) for its pins in a library called Pin2Vec, and [Instacart has embeddings](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc) for its grocery items, stores, and customers. Read more about re-using categorical embeddings in [Rachel's article](https://www.fast.ai/2018/04/29/categorical-embeddings).

![Visualisation of embedding vectors](/images/instacart.png "From Instacart's Deep Learning with Emojis (not Math)")

We can now replace the categorical variables with this trained embeddings in our Random forests or LGBM model. The paper showed that we achieve better accuracy by using entity embeddings.

![Reusing embeddings](/images/reusing-embeddings.png "Results comparison by replacing categorical variables with Entity Embeddings")

After learning this embeddings, we can run PCA or use a t-SNE to project what we learnt for each metadata into 2D dimensional spaces and this can gave us a nice visual idea of the way each metadata value influences the prediction (if one wants to quantitatively assess the importance of a single metadata, one could train a model with only this specific metadata as input).

#### Visualising using t-SNE plots
![Reusing embeddings](/images/qhour_of_day_W.png "T-SNE 2D PROJECTION OF THE EMBEDDINGS LEARNT FOR THE QUARTER OF HOUR AT WHICH THE TAXI DEPARTED (THERE ARE 96 QUARTERS IN ONE DAY, SO 96 POINTS, EACH ONE REPRESENTING A PARTICULAR QUARTER). THIS SUGGESTS THAT EACH QUARTER IS PRETTY IMPORTANT ON ITS OWN.")


### What about continuous features?
Thumb rule is all floating point values in features are continuous. Like card transaction price, time elapsed, distance etc. are considered continuous features. We can try to represent continuous features as categorical depending on the cardinality. But viceversa isn't advised. These continuous features are fed to the model as it is.

### PyTorch model
We will be using PyTorch to build the model. You can use Keras/Tensorflow as well. I'm skipping the data loader, dataset & feature processing part here and made it available as Jupyter notebook [here](https://github.com/chandureddivari/kaggle/blob/master/Pytorch%20data%20loader%20%26%20category%20embeddings.ipynb). There are lot of hyper parameters to finetune like like number of hidden linear layers, dropouts, embedding sizes & dropouts etc. Checkout out [Sacred](https://github.com/IDSIA/sacred) to keep track of your experiments using different hyper parameters.

```python
class TabularModel(nn.Module):
    def __init__(self, emb_sizes, emb_dropout, lin_layers, lin_layers_dropout, n_cat_fields, n_cont_fields, y_range):
        super().__init__()
        # get embeddings
        self.embeddings = get_embeddings(emb_sizes)
        # embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        # calculate linear layer sizes accounting embeddings
        emb_vectors_sum = sum([e.embedding_dim for e in self.embeddings])
        
        # Linear layer sizes are sum of embeddings size + contiguous fields' size + linear layers we wish to have
        linear_szs = [emb_vectors_sum + n_cont_fields] + lin_layers
        
        self.n_cont_fields = n_cont_fields
        # initialize linear layers
        self.lin_layers = nn.ModuleList([nn.Linear(linear_szs[i], linear_szs[i+1]) 
                                         for i in range(len(linear_szs)-1)])
        # Define output layer
        self.output_layer = nn.Linear(linear_szs[-1], 1)
        
        # Initialize batch normalisation for linear layers
        self.batch_norms_lin = nn.ModuleList([nn.BatchNorm1d(s) for s in linear_szs[1:]])
        # Initialize batch normalisation for continous fields
        self.batch_norm_cont = nn.BatchNorm1d(n_cont_fields)
        
        # dropout for linear layers
        self.linear_drops = nn.ModuleList([nn.Dropout(p) for p in lin_layers_dropout])
        
        self.y_range = y_range
        
    def forward(self, cat_fields, cont_fields):
        # Initialize embeddings for respective categorical fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embeddings)]
        # concatenate all the embeddings on axis 1
        x1 = torch.cat(x1, 1)
        # apply dropout on embeddings
        x1 = self.emb_dropout(x1)
        
        # apply batch normalization on continous fields
        x2 = self.batch_norm_cont(cont_fields)
        
        # concatenate along axis 1
        x1 = torch.cat([x1,x2], 1)
        
        # apply linear layers and respective batch norms followed by dropouts 
        for lin, drop, bn in zip(self.lin_layers, self.linear_drops, self.batch_norms_lin):
            # Non linear activation function relu will give only the non-negative values, negatives zeroed.
            x1 = F.relu(lin(x1))
            x1 = bn(x1)
            x1 = drop(x1)
        x1 = self.output_layer(x1)
        # pass the final layer through sigmoid which gives a value between 0 & 1
        x1 = torch.sigmoid(x1)
        y_min = self.y_range[0]
        y_max = self.y_range[1]
        # Mulitply/scale the output from sigmoid with the range of target to get our required y value.
        x1 = x1*(y_max-y_min)
        x1 = y_min + x1
        
        return x1
```

### What next?
Embeddings are becoming ubiquitous. They're also used for colloborative filtering in recommendation engines. PCA on trained embeddings gave interesting results on [movie recommendation dataset](https://towardsdatascience.com/fast-ai-season-1-episode-5-1-movie-recommendation-using-fastai-a53ed8e41269). 

I came across [Kaggle Pet finder adoption competition](https://www.kaggle.com/c/petfinder-adoption-prediction/data). It's quite interesting as the objective is to find out how fast the pet will be adopted based on text, tabular, and image data. Maybe we can leverage the trained categorical embeddings of tabular data by plugging them to a seq-2-seq model along with image & text embeddings. That's my weekend project. Let me know over [twitter](https://twitter.com/chandureddivari) if you've come across any papers or creative ways of using categorical embeddings of tabular data. 

You can find the complete jupyter notebook for the above article on my [github](https://github.com/chandureddivari/kaggle/blob/master/Pytorch%20data%20loader%20%26%20category%20embeddings.ipynb).

Acknowledgements:

Thanks to Jeremy Howard & Rachel from [fast.ai](https://www.fast.ai) for the lesson on using entity embeddings on categorical variables & many other wonderful videos on Machine learning & Deep learning. I sincerely admire your efforts to democratize ML & AI.

### References:

1. [An Introduction to Deep Learning for Tabular Data](https://www.fast.ai/2018/04/29/categorical-embeddings/)
2. [Taxi Trajectory Winners' Interview: 1st place, Team ](http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/)
3. [Rossmann Store Sales, Winner's Interview: 3rd place, Neokami Inc.](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/)
4. [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)

