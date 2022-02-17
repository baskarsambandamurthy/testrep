

# JOB-A-THON - February 2022
## Score
- Private LB Rank:  
- Private LB Score:  
- Public LB Rank: 55th
- Public LB Score:  0.4256


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
      subgraph py3[Tune Model]
        py31[create_model]
        py32[tune_model for 100 iterations]
      end    
      subgraph py4[Final Model]
        py41[(Train Set Full)]
        py42[finalize_model]
        py43[test predictions]
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
