

# JOB-A-THON - February 2022
## Score
- Private LB Rank:  
- Private LB Score:  
- Public LB Rank: 55th
- Public LB Score:  0.4256


```mermaid
flowchart LR
    modelset2-->py21
    modelset1-->py22
    py21 --> py22
    py3 -->|tuned parameters|py22
    subgraph py[Model using pycaret]
      subgraph py1[Initialize pycaret]
        py11[setup]
      end
      subgraph modelset[Model Settings]
        subgraph modelset1[ML algorithm]
          modelset11[XGBoost]
        end
        subgraph modelset2[Validation Settings]
          modelset21[KFold]
          modelset22[No of Folds =10]
        end
      end
      subgraph py2[Evaluate Model]
        subgraph py21[Train Validation Split]
          py211[(Train Set)]
          py212[(Validation Set)]
         end
         py22[[Train Model]]
       end
      subgraph py3[Tune Model]
        py31[create_model]
        py32[tune_model for 100 iterations]
      end    
      subgraph py4[Final Model]
        py41[finalize_model]
        py42[test predictions]
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
