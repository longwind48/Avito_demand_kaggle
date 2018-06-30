# Avito Demand Prediction Challenge

by: Traci

---

The markdown document highlights a detailed reflection of my experience in my first Kaggle competition, [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction/overview). 

It was a challenge posed by Russia's largest classified advertisement website, where participants are required to predict the demand for an online advertisement based on a variety of data: images, text, geospatial information, unlabeled additional data, numeric data, categorical data. It is essentially a low-key time series problem, as the test set instances were in April and train set instances were predominantly in March. 

My submission managed to get into the **top 12%** among 1917 kagglers. Although I was just 0.0002 short of the top 10% score, It was very refreshing and challenging to come up with a decent submission in under 20 days. 

---



##My approach:

Exploratory data analysis is always crucial in any machine learning process, it allows one to get closer to the certainty that future results will be valid, correctly interpreted, and applicable to business contexts ([a good 5 minute read on EDA](https://indatalabs.com/blog/data-science/datascience-project-exploratory-data-analysis#BYXceB0zuh9F58m4.99)). Thanks to the generosity of Kaggle community,  [good EDA scripts](https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito) are easy to come by. 

Due to my late entry to the competition, my code base was from [Benjamin](https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm) and [Himanshu](https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0). Thanks! Plenty of feature engineering efforts were made after some level of domain research and ask-arounds, however, the main challenge I faced were CPU and memory bottlenecks. Working with a 4-core and 16GB-ram laptop definitely frustrating as my ipython notebook kept crashing. I had to resort to renting a 8-core and 32Gb-ram cloud machine from paperspace. 

I primarily focused on optimizing my Lightgbm model because of its speed. XGBoost and CatBoost were also experimented. The downside of CatBoost is that it doesn't support sparse matrices as input; because I used tdidf to handle my text features, it outputs a sparse matrix, which can't be used directly by CatBoost. I used truncated SVD to 'convert' the tdidf sparse matrix into `dataframe` form. Although I only used 10 components out of 1.4mil possible columns (to save time), they still explained 9% of the variance, which was amazing! Yandex's CatBoost algorithm deals with categorical values intelligently by transforming them in numeric values using the following formula:
$$
avgTarget = \frac{countInClass + prior}{totalCount + 1}
$$
More details [here](https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/). Although it was documented that CatBoost has better performance than Lightgbm and XGBoost, in this competition, Lightgbm still trumped in terms of performance, probably because the edge CatBoost has over other boosting algorithms is very dependent on the type of data. Whichever the case, it is still hard to estimate the true reason of subpar performance for CatBoost models, since XGboost, Lightgbm and CatBoost all grow and prune their trees differently, and also have different limitations and hyperparameters. 

I played around with target encodings (aka mean encodings) because the data has some high cardinality categorical features, and the downsides of tree-boosting algorithms is its inability to handle them optimally. In general, the more complicated and non-linear feature target dependency is, the more effective is target encoding. Good indicators of implementing target encoding can be:

1. Presence of high cardinality features
2. When increasing the depth of tree-boosting models, training and validation metrics naturally improves, showing no sign of overfitting. This is a sign that trees need a huge number of splits to extract information from some variables.

Let's dive into my favourite part of this reflection.

---

## Lessons learnt

- Although categorical feature interactions were experimented,  it actually worsened the rmse score. I supposed feature interactions makes overfitting easier, despite enlarging the feature space which makes learning easier. Tree-based models can really benefit from feature interactions because it is difficult to extract such dependencies without using this transformation. My mistake was I did not apply any feature selection techniques to fish out those problem features. The [7th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/60026) mentioned that three-way interactions didn't work.
- Numeric feature interactions between the most predictive features could boost my score even more. 

- It is important to mimic the train/test split with a proper time-based validation split for a less biased relationship between private LB score and CV score. Multiple folds cross validation implementation for time-series train/validation split could be implemented as follows.

  ```python
  # Credits to https://www.kaggle.com/c/avito-demand-prediction/discussion/59914
  
  # Seperate dates from train set into 5 groups
  foldls = [["2017-03-15", "2017-03-16", "2017-03-17"], \
         ["2017-03-18", "2017-03-19", "2017-03-20"], \
         ["2017-03-21", "2017-03-22", "2017-03-23"], \
         ["2017-03-24", "2017-03-25", "2017-03-26"], \
          ["2017-03-27", "2017-03-28", "2017-03-29", \
              "2017-03-30", "2017-03-31", "2017-04-01", \
              "2017-04-02", "2017-04-03","2017-04-07"]]
  
  # Convert them into datetime format
  foldls = [[pd.to_datetime(d) for d in f] for f in foldls]
  
  # Create a new column in dataframe with -1's
  df['fold'] = -1
  
  # Assign each instance their corresponding fold they belong to 
  for t, fold in enumerate(foldls):
      df['fold'][df.activation_date.isin(fold)] = t
      
  # Now all instances in train set will have a new feature indicating which fold they belong to and instances in test set will be assigned fold -1
  df['fold'].value_counts()
  
  ```

- Bayesian Optimization was my main method of tuning Lightgbm and XGBoost models. Once again, because my cross validation approach was not set up to mimic the train/test split, the optimized parameters were not accurate. Fitting the optimized parameters on the entire train test actually gave a poorer score.

  To put in simplified terms, underfitting is when your training metric can't improve anymore, which can be attributed to high bias, something that causes an algorithm to miss the relevant relations between features and target. High bias algorithms also produce simpler models that don't tend to overfit but may *underfit* their training data, failing to capture important regularities. We know bias as the difference between the expected prediction of our model and the correct value which we are trying to predict. I failed to detect signs of underfitting because I was too wary of overfitting. 

  |                     | learning rate | training rmse | validation rmse | private LB rmse | Diff (train and LB rmse) |
  | ------------------- | ------------- | ------------- | --------------- | --------------- | ------------------------ |
  | w/o target encoding | 0.016         | 0.19099       | 0.21676         | 0.2215          | 0.0305                   |
  | w/ target encoding  | 0.016         | 0.20025       | 0.21634         | 0.2229          | 0.0227                   |

  We can observe here that although it seemed like the model built w/o target encoding has a higher bias than w/ target encoding (due to higher difference between train and LB rmse), both models are underfitting the data because of the large difference between training and validation. Furthermore, validation rmse seemed to have improved for the case of target encoding, but it actually worsened the model's ability to generalize the true test set. This is purely the problem of incorrect train/validation split.

  Overfitting, on the other hand, is when your training score is going down continuously while your test or validation score is going up continuously. This is because the model is too complex and suffer from high variance. The variance is how much the predictions for a given point vary between different realizations of the model. 

  I made several huge blunders because I did not access the difference between training rmse's. After reading a godsend article by [Scott Fortmann-Roe](http://scott.fortmann-roe.com/docs/BiasVariance.html), I have learnt to minimize bias even at the expense of variance. Yes, high variance is also bad, but a model with high variance could at least predict well on average, which is not *fundamentally wrong*. 

- Better manage my memory usage to avoid bottlenecks.

  ```python
  def reduce_memory(df):
      for c in df.columns:
          if df[c].dtype=='int':
              if df[c].min()<0:
                  if df[c].abs().max()<2**7:
                      df[c] = df[c].astype('int8')
                  elif df[c].abs().max()<2**15:
                      df[c] = df[c].astype('int16')
                  elif df[c].abs().max()<2**31:
                      df[c] = df[c].astype('int32')
                  else:
                      continue
              else:
                  if df[c].max()<2**8:
                      df[c] = df[c].astype('uint8')
                  elif df[c].max()<2**16:
                      df[c] = df[c].astype('uint16')
                  elif df[c].max()<2**32:
                      df[c] = df[c].astype('uint32')
                  else:
                      continue
      return df
  ```

- Use `hashingvectorizer`rather than straight-up tf-idf is significantly more memory efficient.

- Mitigate the computationally demanding problem of using a high `n_features` parameter with truncated singular value decomposition (tSVD). (200-300 components will do fine). My mistake was that I kept my text engineering process static, because of memory bottleneck issues when I experimented with char ngrams and a higher `ngram_range`. 

```python
# Credits to https://www.kaggle.com/c/avito-demand-prediction/discussion/59881
hv = HashingVectorizer(stop_words=stopwords.words('russian'), 
                     lowercase=True, ngram_range=(1, 3),
                     n_features=200000)
hv_feats = hv.fit_transform(user_text_df['all_text'])

tfidf_user_text = TfidfTransformer() 

tfidf_user_text_feats = tfidf_user_text.fit_transform(hv_feats)
```

- I could be more bold with my parameter selection for Lightgbm model. Instead of worrying about overfitting, a lower `learning_rate` and higher value of `num_leaves` should always be considered. Failing to experiment with this caused me to fall short of the top 10% mark! The [5th place solution](https://github.com/darraghdog/avito-demand/) had their learning rate set to 0.01 and leaves set to 1000. 
- My implementation of target encoding still suffered from a sizable amount of underfitting, achieving a private LB score of 0.2229. I didn't build a better suited cross-validation split that mimic the train/test split. Also, I could have explored different presets of the prior weight function. the [86th place solution](https://www.kaggle.com/johnfarrell/adp-prepare-data-me-20-2-true) implemented   [target encoding](https://www.kaggle.com/c/avito-demand-prediction/discussion/59895) using a combination of data from other CV folds and strictly past data. 
- Proper version control should be set up. Irregular naming of scripts and notebooks caused me so much trouble. :cry: 
- My feature engineering efforts could be improved with a concrete plan and proper documentation. Although I consistently tracked my progress and planned my approach in google docs, It was still unorganized. I really appreciate the advice posted by [Joe](https://www.kaggle.com/c/avito-demand-prediction/discussion/56986#330023), who wrote a concise feature engineering approach. 
- Always start early in the competition! I started 20 days before the competition ends, which is obviously not enough to go through a competitive machine learning process. What was I thinking? :sweat_smile:
- I was able to implement a simple weighted ensemble of my CatBoost, Lightgbm and XGBoost models. I understood the benefits of stacking and how it could empower the model, but designing multi-layered stacking seems too much of a drag to me. I rather use the time to play with some other projects. No offence to the stacking gods. Maybe next time.

---

### To sum it up...

The key takeaway here is that I still have much to learn, and I think, this is the beauty of machine learning. It is truly amazing to be able to translate trails and evidences of activity from the public to something potentially useful. Can you imagine a number popping up to let you know how likely your product is going to be sold when you are trying to sell a pair of shoes on Avito? :joy: 

This project is really interesting and refreshing!

---

## Introducing my teachers:

- [3th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59885) 
- [4th Place Solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59881) 
- [5th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59914) [[repo](https://github.com/darraghdog/avito-demand/)]
- [7th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/60026) 
- [14th Place Solution: The Almost Golden Defenders](https://www.kaggle.com/c/avito-demand-prediction/discussion/60059) 
- [25th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59902) 
- [86th place script on target encoding](https://www.kaggle.com/c/avito-demand-prediction/discussion/59895)

- [How to Select Your Final Models in a Kaggle Competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)
- [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html) 
