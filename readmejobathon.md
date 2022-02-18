# JOB-A-THON - February 2022
## Score
- Private LB Rank:  
- Private LB Score:  
- Public LB Rank: 55th
- Public LB Score:  0.4256

## Key points for top score achievement

1. Understanding of high cardinality of user_id field
2. Test and Train similarity differences from EDA
3. Target Encoding of the user_id field
4. Tuning parameters of Gradient Boosting algorithm XGBoost
5. Test Predictions using full train data set with the iterations 

## High Level Design
```mermaid
flowchart TB
    FE --> py
    subgraph py[Model using pycaret]
      direction TB
      subgraph py1[Initialize pycaret]
        py11[setup]
      end
      modelset --> py2
      py1 --> modelset
      py2 --> sel{Select Model}
      sel --> py3
      py3 -->|tuned parameters|py4
      subgraph py2[Evaluate Model]
          subgraph py21[Train Validation Split]
            py211[(Train Set)]
            py212[(Validation Set)]
          end
          subgraph py22[Train Models]
            py221[[Train Model]]
            py222[[Train Model]]
            py223[[Train Model]]
            py224[[Train Model]]
          end
      end
      subgraph py3[Tune Model - XGBoost]
        py31[Train Model] --> py32[Tune Model for 100 iterations]
      end    
      subgraph py4[Final Model]
        py41[(Train Set Full)]
        py42[Train Model] --> py43[Test predictions]
      end       
      subgraph modelset[Model Settings]
        subgraph modelset1[ML algorithm]
          modelset11[LightGBM]
          modelset12[XGBoost]
          modelset13[CatBoost]
          modelset14[Random Forest]
        end

        subgraph modelset2[Validation Settings]
          modelset21[KFold]
          modelset22[No of Folds =10]
        end
        subgraph modelsette[Target Encoding]
          modelsette1[user_id target enc]
          modelsette2[video_id target enc]
          modelsette3[category_id target enc]
        end
      end
    end
    subgraph FE[Feature Engineering]
    direction TB
    subgraph SFE[Simple]
        direction LR
        LE[Label Encoding]
        OE[Ordinal Encoding]
    end
    subgraph ADV[Advanced Feature Engineering]
        direction LR
        AGG1[user_id aggregation]
        AGG2[category_id aggregation]
    end
  end
  subgraph EDA
    EDA1[pandas profiling]
  end
  EDA -->FE 
````
## EDA
 -  For EDA, package `pandas_profiling` is used which provides descriptive statistics, histogram, missing values, duplicates for each of the feature in the given dataset.
 -  Profile summary has been generated for both train and test data.

code to generate EDA using pandas profiling<br>
```python
import pandas_profiling as pp
pp.ProfileReport(train)
```

- Findings are below:

| Findings|  Suggestions
|----------------------|-------------------------------|
| There are no missing values in any of the columns        | Missing value imputation is not needed
| Target column engagement_score contains float values | So training objective should be regression
| Numeric columns are age, followers, views | During model training, these columns to be considered as numerical types 
| categorical columns are user_id, video_id, category_id, profession and gender | These columns require transformations so that it could be effectively used by the machine learning algorithms. 
|  profession and gender have text values | These columns have to be transformed with numeric values. 
| user_id column contains 27000+ unique values         | So high cardinality treatment need to be performed on user_id column
|All user_ids in test set exist in the training set.        | So user_id in training would play important role during training
| Train set contains list of specific user and video combinations and the combinations of video id and user id in train set   | The user_id and video_id combination would not have any effect on model training.

## Feature Engineering

### Simple Feature Engineering

- Label encoding is performed on the categorical data `gender`
- Feature `profession` is ordinal data type and its labels are encoded using ordered set of values

### Advanced Feature Engineering

#### Target Encoding
- The `user_id` categorical field has high cardinality (more than 20000 unique values in train) and if this categorical field is converted using one hot encoding, it would generate around 20000 features and this would consume huge memory and cpu. The optimaly techinique to utilize this field is to apply target encoding on this field.
- Target Encoding should not be performed before cross validation and it has been performed during cross validation.
> **Note**: If Target Encoding is performed before cross validation (i.e) performed for entire train set and then if the cross validation is performed using such encoding, then it would result in target leak which means that the partial train set of each fold has got the target leak of the validation set. This would result in very good validation score and poor test score.
- As pycaret package is used for model training and cross validation, the target encoding is mentioned as custom_pipeline parameter for the pycaret setup.
> **Note**: `ColumnTransformer()` is used as pipeline transformer on the columns to transform target encoding and `TargetEncoder()` is used as the actual transformer for target encoding.
- Features `video_id` and `category_id` are also converted using target encoding

#### Feature Aggregations

- User Groupings: Count of videos and Count of categories grouped per user are generated. 
- Category Groupings: Count of users and Count of videos grouped per category are generated. 

List of feature aggregations are as below

| Feature Name |  Grouping By | Description
|----------------------|-------------------------------|-------------------------------|
| user_video_count   | user_id | Count of videos grouped per user
| user_category_count      | user_id |Count of categories grouped per user
| category_user_count      | category_id |Count of users grouped per category
| category_video_count      | category_id |Count of videos grouped per category

#### Usage of original user_id

Original user_id field is also used as a feature besides the target encoded value of the user_id field

## Model Build - Train - Predict

### Process

1. The below models are evaluated using 10 fold cross validation and `KFold` technique is used. Here fixed number of estimators are used.
    - LightGBM
    - XGBoost
    - CatBoost
    - Random Forest     
2. One of the above Machine Learning algorithm is selected based on the criteria of model execution speed and the predictability on GPU device.
3. In this dataset, XGBoost model has been selected since it is 4 times faster than other boosting models when executed in GPU compared to others while at the same time it yields almost similar predictability as other boosting models like LightGBM and Catboost.
4. Evaluated XGBoost model is tuned for optimal parameters for which the validation score yields better results.
5. Then Model is trained using full training set with the tuned model parameters and number of estimators. This model is called `Final Model`
> **Note**: In the `Final Model`, full training set is used which is different from cross validation where only partial training set is used for each fold.
6. Predictions of the Test set are performed using the trained `Final Model`.

### Initialize
  - Initializes training and target data in the pycaret package. Note that the target encoding process to be specified as `custom_pipeline` argument in the pycaret initialization. When machine supports gpu, set gpu flag to speed up execution of the machine learning algorithms. <br>
pycaret code:<br>
<br>

code to create pipeline for target encoding to be used during cross validation<br>
```python
ct = ColumnTransformer(
     [
         ("targetenc",  TargetEncoder(cols=['user_id','category_id', 'video_id'],
                                             min_samples_leaf=2, smoothing=0.1) , 
                                             ['user_id','category_id', 'video_id']),
      ],remainder='passthrough')
preprocessor = ('preproc',ct)  
```

code to initialize
```python
  
setup(     train[features], 
            session_id=100,
            ....
          numeric_features=numeric_cols,
          target = targetcol,
          custom_pipeline=preprocessor,
          fold=10,fold_shuffle=True,
          use_gpu=True,
          ....
          
        )
```

### Evaluate Model
In the implementation, 4 models are evaluated using the split train and validation datasets with 10 folds.Among the models, there are 3 boosting algorithms and 1 bagging algorithm.
<br>
code to train LightGBM using pycaret:<br>
```python
lgbm = create_model('lightgbm')
```
code to train XGBoost using pycaret:<br>
```python
xgbm = create_model('xgboost')
```
code to train Catboost using pycaret:<br>
```python
catboost = create_model('catboost')
```
code to train Random Forest using pycaret:<br>
```python
rf = create_model('rf')
```

### Select Model
 - Based on the results from the training of the above 4 models, xgboost is much faster when executed in GPU and while at the same time, it yields almost same prediction accuracy compared to other models. Hence the **XGBoost** model is selected for further tuning. Since the model tuning would take longer time as it need to train the model for different parameter combinations and hence the criteria for model selection gives slightly higher importance to execution speed.
  
### Tune Model
 - The selected `XGBoost` model training is repeated for around 100 times with different set of parameters. 
 - The parameter optimization is done by using pycaret separate api and using the scikit-optimize package and bayesian technique. The bayesian technique results in faster convergence of optimal results due to its past learning from parameter values compared to random parameter selection.

pycaret code to tune XGBoost :<br>
```python
tuned_xgbm = tune_model('xgboost',
                        search_library='scikit-optimize',
                        n_iter=100)
``` 

### Final Model
 -  In this step, the XGBoost model is trained using full training set with its tuned model parameters and number of estimators. 
 -  Then the final model is used to perform predictions on the test dataset

code for final model using pycaret :<br>
```python
final_model = finalize_model(tuned_xgbm)
```

code for test predictions using pycaret  :<br>
```python
test_preds = predict_model(final_model, data=test)
```
 
 




