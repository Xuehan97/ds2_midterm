Report
================
Xuehan Yang
2022/3/24

``` r
library(tidyverse)
library(caret)
library(pROC)
library(pdp)
library(vip)
library(AppliedPredictiveModeling)
```

# Introduction

Heart failure is a common form of cardiovascular disease, which is a
severe threat to patient’s life. It occurs when the heart muscle doesn’t
pump blood as well as it should. Our research aims predict the death
probability among heart failure patients with machine learning models on
clinical features.

# Data

``` r
hf_df <- read_csv("heart_failure_clinical_records_dataset.csv") %>% janitor::clean_names() %>% mutate(
  death_event = case_when(
    death_event == 1 ~ "Y",
    death_event == 0 ~ "N"
  ),
  death_event = factor(death_event)
)
```

The dataset we analyzed contains the medical records of 299 heart
failure patients collected at the Faisalabad Institute of Cardiology and
at the Allied Hospital in Faisalabad (Punjab, Pakistan), during
April–December 2015 <sup>1</sup> <sup>2</sup>. It is a prospective
cohort study. The patients consisted of 105 women and 194 men, and their
ages range between 40 and 95 years old. The features of those patients
include 13, which further will become our predictors.

# Exploratory analysis/visualization

## Continuous predictors

``` r
featurePlot(x = hf_df[,c(1,3,5,7,8,9,12)],
            y = hf_df$death_event,
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")),
            plot = "density", pch = "l",
            auto.key = list(columns = 2))
```

![](midterm_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Among all the continuous predictors, we can see that death events
grouped when the follow-up-period is short(time &lt; 50), patients with
higher level of serum sodium tended to have death event, and patients
with lower ejection fraction tended to have death event.

## Binary predictors

``` r
contin_tb <- 
hf_df %>% select(anaemia, diabetes, high_blood_pressure, sex, smoking, death_event) %>% 
  pivot_longer(
    cols = 1:5,
    names_to = "feature",
    values_to = "exposed"
  ) %>% 
  group_by(feature, exposed, death_event) %>% 
  summarise(n = n()) %>% 
  pivot_wider(
    names_from = death_event,
    values_from = n
  )

contin_tb %>% knitr::kable()
```

| feature               | exposed |   N |   Y |
|:----------------------|--------:|----:|----:|
| anaemia               |       0 | 120 |  50 |
| anaemia               |       1 |  83 |  46 |
| diabetes              |       0 | 118 |  56 |
| diabetes              |       1 |  85 |  40 |
| high\_blood\_pressure |       0 | 137 |  57 |
| high\_blood\_pressure |       1 |  66 |  39 |
| sex                   |       0 |  71 |  34 |
| sex                   |       1 | 132 |  62 |
| smoking               |       0 | 137 |  66 |
| smoking               |       1 |  66 |  30 |

``` r
contin_tb %>% mutate(risk = Y/N) %>% 
  select(-N,-Y) %>% 
  mutate(
    exposed = case_when(
      exposed == 1 ~ "exposed",
      exposed == 0 ~ "nonexposed"
    )
  ) %>% 
  pivot_wider(
    names_from = exposed,
    values_from = risk
  ) %>% 
  mutate(riskratio = exposed/nonexposed) %>% 
  arrange(desc(riskratio)) %>% knitr::kable()
```

| feature               | nonexposed |   exposed | riskratio |
|:----------------------|-----------:|----------:|----------:|
| high\_blood\_pressure |  0.4160584 | 0.5909091 | 1.4202552 |
| anaemia               |  0.4166667 | 0.5542169 | 1.3301205 |
| diabetes              |  0.4745763 | 0.4705882 | 0.9915966 |
| sex                   |  0.4788732 | 0.4696970 | 0.9808378 |
| smoking               |  0.4817518 | 0.4545455 | 0.9435262 |

Among the binary predictors, the risk ratio of high blood pressure and
anaemia are 1.42 and 1.33, which means the risk of death among heart
failure patients with hypertension is 1.42 times the risk of death among
heart failure patients without hypertension, and the risk of death among
heart failure patients with anaemia is 1.33 times the risk of death
among heart failure patients without anaemia.

# Models

In order to decide whether a patient with heart failure would die in
following period, we use classification models to see what kinds of
features correspond to the death event.

## predictor

There are 12 predictors included in our models, consisting of 7
continuous predictors and 5 binary predictors. Specifically, age of the
patient (age), if decrease of red blood cells or hemoglobin (anaemia),
if the patient has hypertension (high\_blood\_pressure), level of the
CPK enzyme in the blood (creatinine\_phosphokinase), if the patient has
diabetes (diabete), percentage of blood leaving the heart at each
contraction (ejection\_fraction), platelets in the blood (platelets),
woman or man (sex), level of serum creatinine in the blood
(serum\_creatinine), level of serum sodium in the blood (serum\_sodium),
if the patient smokes or not (smoking), follow-up period (time), if the
patient deceased during the follow-up period (death\_event).

## Cross Validation splitting

``` r
set.seed(11)
trainrows <- createDataPartition(y = hf_df$death_event,
                                 p = 0.7,
                                 list = FALSE)
```

Full dataset is partitioned to 70% training data and 30% test data.

## Logistic Regression

$$log(\\frac{\\pi\_i}{1-\\pi\_i}) = \\beta\_0 + \\beta\_1x\_1 + \\beta\_2x\_2 + \\cdots + \\beta\_px\_p\\quad p=12$$

``` r
ctrl <- trainControl(method = "repeatedcv", repeats = 5, number = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

set.seed(22)
glm.model <- train(x = hf_df[trainrows, -13],
                   y = hf_df$death_event[trainrows],
                   method = "glm",
                   metric = "ROC",
                   trControl = ctrl)
summary(glm.model)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.2338  -0.5755  -0.2225   0.4360   2.6072  
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)               1.736e+01  7.820e+00   2.220  0.02639 *  
    ## age                       5.184e-02  1.890e-02   2.743  0.00609 ** 
    ## anaemia                  -2.552e-02  4.311e-01  -0.059  0.95279    
    ## creatinine_phosphokinase  1.656e-04  2.006e-04   0.826  0.40898    
    ## diabetes                  1.177e-01  4.183e-01   0.281  0.77844    
    ## ejection_fraction        -9.260e-02  2.164e-02  -4.279 1.88e-05 ***
    ## high_blood_pressure      -4.706e-01  4.611e-01  -1.021  0.30739    
    ## platelets                 1.930e-07  2.435e-06   0.079  0.93681    
    ## serum_creatinine          2.269e-01  2.485e-01   0.913  0.36136    
    ## serum_sodium             -1.133e-01  5.597e-02  -2.024  0.04292 *  
    ## sex                      -7.019e-01  5.126e-01  -1.369  0.17088    
    ## smoking                  -1.417e-01  5.106e-01  -0.278  0.78138    
    ## time                     -2.177e-02  3.662e-03  -5.945 2.77e-09 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 265.26  on 210  degrees of freedom
    ## Residual deviance: 153.72  on 198  degrees of freedom
    ## AIC: 179.72
    ## 
    ## Number of Fisher Scoring iterations: 6

## Penalized Logistic Regression

``` r
glmnGrid <- expand.grid(.alpha = seq(0, 1, length = 21), .lambda = exp(seq(-5, 2, length = 40)))

set.seed(22)
glmn.model <- train(x = hf_df[trainrows, -13],
                    y = hf_df$death_event[trainrows],
                    method = "glmnet",
                    tuneGrid = glmnGrid,
                    metric = "ROC",
                    trControl = ctrl)
glmn.model$bestTune
```

    ##    alpha    lambda
    ## 68  0.05 0.8574039

## Logistic with GAM

``` r
set.seed(22)
gam.model <- train(x = hf_df[trainrows, -13],
                   y = hf_df$death_event[trainrows],
                   method = "gam",
                   metric = "ROC",
                   trControl = ctrl)
gam.model$finalModel
```

    ## 
    ## Family: binomial 
    ## Link function: logit 
    ## 
    ## Formula:
    ## .outcome ~ anaemia + diabetes + high_blood_pressure + sex + smoking + 
    ##     s(ejection_fraction) + s(serum_sodium) + s(serum_creatinine) + 
    ##     s(age) + s(time) + s(platelets) + s(creatinine_phosphokinase)
    ## 
    ## Estimated degrees of freedom:
    ## 1.46 1.00 2.13 1.00 9.00 3.58 7.90 
    ##  total = 32.06 
    ## 
    ## UBRE score: -0.3571576

``` r
summary(gam.model)
```

    ## 
    ## Family: binomial 
    ## Link function: logit 
    ## 
    ## Formula:
    ## .outcome ~ anaemia + diabetes + high_blood_pressure + sex + smoking + 
    ##     s(ejection_fraction) + s(serum_sodium) + s(serum_creatinine) + 
    ##     s(age) + s(time) + s(platelets) + s(creatinine_phosphokinase)
    ## 
    ## Parametric coefficients:
    ##                     Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)         -14.8754    41.2918  -0.360   0.7187  
    ## anaemia              -1.1052     0.7149  -1.546   0.1221  
    ## diabetes             -1.0662     0.7414  -1.438   0.1504  
    ## high_blood_pressure  -0.4843     0.7800  -0.621   0.5347  
    ## sex                  -1.3565     0.8165  -1.661   0.0967 .
    ## smoking              -0.3730     0.8696  -0.429   0.6680  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Approximate significance of smooth terms:
    ##                               edf Ref.df Chi.sq  p-value    
    ## s(ejection_fraction)        1.457  1.795 10.806 0.002560 ** 
    ## s(serum_sodium)             1.000  1.000  3.263 0.070880 .  
    ## s(serum_creatinine)         2.127  2.681 10.039 0.018123 *  
    ## s(age)                      1.000  1.000  8.468 0.003614 ** 
    ## s(time)                     9.000  9.000 29.393 0.000552 ***
    ## s(platelets)                3.580  4.420 10.168 0.060791 .  
    ## s(creatinine_phosphokinase) 7.898  7.992 13.688 0.091188 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## R-sq.(adj) =   0.73   Deviance explained =   73%
    ## UBRE = -0.35716  Scale est. = 1         n = 211

## Logistic with MARS to add interaction

``` r
set.seed(22)
mars.model <- train(x = hf_df[trainrows,-13],
                    y = hf_df$death_event[trainrows],
                    method = "earth",
                    tuneGrid = expand.grid(degree = 1:4, nprune = 2:20),
                    metric = "ROC",
                    trControl = ctrl)
plot(mars.model)
```

![](midterm_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
summary(mars.model)
```

    ## Call: earth(x=tbl_df[211,12], y=factor.object, keepxy=TRUE,
    ##             glm=list(family=function.object, maxit=100), degree=1, nprune=8)
    ## 
    ## GLM coefficients
    ##                                   Y
    ## (Intercept)               1.6160105
    ## h(age-67)                 0.1254412
    ## h(35-ejection_fraction)   0.1844068
    ## h(1.83-serum_creatinine) -2.0269340
    ## h(time-64)               -0.2397973
    ## h(time-80)                0.2690722
    ## h(time-170)              -0.1331508
    ## h(time-210)               0.1261266
    ## 
    ## GLM (family binomial, link logit):
    ##  nulldev  df       dev  df   devratio     AIC iters converged
    ##  265.257 210    125.66 203      0.526   141.7     6         1
    ## 
    ## Earth selected 8 of 19 terms, and 4 of 12 predictors (nprune=8)
    ## Termination condition: Reached nk 25
    ## Importance: time, ejection_fraction, serum_creatinine, age, ...
    ## Number of terms at each degree of interaction: 1 7 (additive model)
    ## Earth GCV 0.1139433    RSS 20.74524    GRSq 0.4832478    RSq 0.5498514

``` r
mars.model$bestTune
```

    ##   nprune degree
    ## 7      8      1

``` r
pdp::partial(mars.model, pred.var = c("age"), grid.resolution = 200) %>% autoplot()
```

    ## Warning: Use of `object[[1L]]` is discouraged. Use `.data[[1L]]` instead.

    ## Warning: Use of `object[["yhat"]]` is discouraged. Use `.data[["yhat"]]`
    ## instead.

![](midterm_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
vip(mars.model$finalModel)
```

![](midterm_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

## LDA

``` r
set.seed(22)
lda.model <- train(x = hf_df[trainrows,-13],
                    y = hf_df$death_event[trainrows],
                    method = "lda",
                    metric = "ROC",
                    trControl = ctrl)
summary(lda.model)
```

    ##             Length Class      Mode     
    ## prior        2     -none-     numeric  
    ## counts       2     -none-     numeric  
    ## means       24     -none-     numeric  
    ## scaling     12     -none-     numeric  
    ## lev          2     -none-     character
    ## svd          1     -none-     numeric  
    ## N            1     -none-     numeric  
    ## call         3     -none-     call     
    ## xNames      12     -none-     character
    ## problemType  1     -none-     character
    ## tuneValue    1     data.frame list     
    ## obsLevels    2     -none-     character
    ## param        0     -none-     list

``` r
lda.model$finalModel
```

    ## Call:
    ## lda(x, y)
    ## 
    ## Prior probabilities of groups:
    ##         N         Y 
    ## 0.6777251 0.3222749 
    ## 
    ## Group means:
    ##        age   anaemia creatinine_phosphokinase  diabetes ejection_fraction
    ## N 59.51515 0.4055944                 596.0699 0.4055944          40.26573
    ## Y 66.14216 0.4558824                 632.6324 0.3970588          31.92647
    ##   high_blood_pressure platelets serum_creatinine serum_sodium       sex
    ## N           0.3146853  265492.3         1.213986     137.4126 0.6293706
    ## Y           0.3970588  254689.2         1.702059     135.0147 0.6617647
    ##     smoking      time
    ## N 0.3286713 157.37063
    ## Y 0.3235294  69.23529
    ## 
    ## Coefficients of linear discriminants:
    ##                                    LD1
    ## age                       2.666305e-02
    ## anaemia                  -4.368933e-02
    ## creatinine_phosphokinase  9.969637e-05
    ## diabetes                  6.129474e-02
    ## ejection_fraction        -4.700353e-02
    ## high_blood_pressure      -2.110655e-01
    ## platelets                 2.661039e-07
    ## serum_creatinine          1.703615e-01
    ## serum_sodium             -4.990688e-02
    ## sex                      -2.774251e-01
    ## smoking                  -8.215755e-02
    ## time                     -1.200406e-02

## Naive Bayes

``` r
nbGrid <- expand.grid(usekernel = c(FALSE,TRUE), fL = 1, adjust = seq(.2, 3, by = .2))

set.seed(22)
nb.model <- train(x = hf_df[trainrows, -13],
                  y = hf_df$death_event[trainrows],
                  method = "nb",
                  tuneGrid = nbGrid,
                  metric = "ROC",
                  trControl = ctrl)
plot(nb.model)
```

![](midterm_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

## Compare test performance

``` r
glm.pred <- predict(glm.model, newdata = hf_df[-trainrows,], type = "prob")[,2]
glmn.pred <- predict(glmn.model, newdata = hf_df[-trainrows,], type = "prob")[,2]
gam.pred <- predict(gam.model, newdata = hf_df[-trainrows,], type = "prob")[,2]
mars.pred <- predict(mars.model, newdata = hf_df[-trainrows,], type = "prob")[,2]
lda.pred <- predict(lda.model, newdata = hf_df[-trainrows,], type = "prob")[,2]
nb.pred <- predict(nb.model, newdata = hf_df[-trainrows,], type = "prob")[,2]

roc.glm <- roc(hf_df$death_event[-trainrows], glm.pred)
roc.glmn <- roc(hf_df$death_event[-trainrows], glmn.pred)
roc.gam <- roc(hf_df$death_event[-trainrows], gam.pred)
roc.mars <- roc(hf_df$death_event[-trainrows], mars.pred)
roc.lda <- roc(hf_df$death_event[-trainrows], lda.pred)
roc.nb <- roc(hf_df$death_event[-trainrows], nb.pred)

auc <- c(roc.glm$auc[1], roc.glmn$auc[1], roc.gam$auc[1], roc.mars$auc[1], roc.lda$auc[1], roc.nb$auc[1])

plot(roc.glm, legacy.axes = TRUE)
plot(roc.glmn, col = 2, add = TRUE)
plot(roc.gam, col = 3, add = TRUE)
plot(roc.mars, col = 4, add = TRUE)
plot(roc.lda, col = 5, add = TRUE)
plot(roc.nb, col = 6, add = TRUE)

modelNames <- c("glm","glmn","gam","mars","lda","nb")
legend("bottomright", legend = paste0(modelNames, ": ", round(auc,3)),
col = 1:6, lwd = 2)
```

![](midterm_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

Confusion Matrix

``` r
test.glm <- rep("N", length(glm.pred))
test.glm[glm.pred > 0.4] <- "Y"
confusionMatrix(data = as.factor(test.glm), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 54 11
    ##          Y  6 17
    ##                                           
    ##                Accuracy : 0.8068          
    ##                  95% CI : (0.7088, 0.8832)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.006377        
    ##                                           
    ##                   Kappa : 0.5325          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.331975        
    ##                                           
    ##             Sensitivity : 0.6071          
    ##             Specificity : 0.9000          
    ##          Pos Pred Value : 0.7391          
    ##          Neg Pred Value : 0.8308          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.1932          
    ##    Detection Prevalence : 0.2614          
    ##       Balanced Accuracy : 0.7536          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.glmn <- rep("N", length(glmn.pred))
test.glmn[glmn.pred > 0.4] <- "Y"
confusionMatrix(data = as.factor(test.glmn), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 60 23
    ##          Y  0  5
    ##                                           
    ##                Accuracy : 0.7386          
    ##                  95% CI : (0.6341, 0.8266)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.1512          
    ##                                           
    ##                   Kappa : 0.2287          
    ##                                           
    ##  Mcnemar's Test P-Value : 4.49e-06        
    ##                                           
    ##             Sensitivity : 0.17857         
    ##             Specificity : 1.00000         
    ##          Pos Pred Value : 1.00000         
    ##          Neg Pred Value : 0.72289         
    ##              Prevalence : 0.31818         
    ##          Detection Rate : 0.05682         
    ##    Detection Prevalence : 0.05682         
    ##       Balanced Accuracy : 0.58929         
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.gam <- rep("N", length(gam.pred))
test.gam[gam.pred > 0.4] <- "Y"
confusionMatrix(data = as.factor(test.gam), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 55 13
    ##          Y  5 15
    ##                                          
    ##                Accuracy : 0.7955         
    ##                  95% CI : (0.6961, 0.874)
    ##     No Information Rate : 0.6818         
    ##     P-Value [Acc > NIR] : 0.01252        
    ##                                          
    ##                   Kappa : 0.4897         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.09896        
    ##                                          
    ##             Sensitivity : 0.5357         
    ##             Specificity : 0.9167         
    ##          Pos Pred Value : 0.7500         
    ##          Neg Pred Value : 0.8088         
    ##              Prevalence : 0.3182         
    ##          Detection Rate : 0.1705         
    ##    Detection Prevalence : 0.2273         
    ##       Balanced Accuracy : 0.7262         
    ##                                          
    ##        'Positive' Class : Y              
    ## 

``` r
test.mars <- rep("N", length(mars.pred))
test.mars[mars.pred > 0.4] <- "Y"
confusionMatrix(data = as.factor(test.mars), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 55  6
    ##          Y  5 22
    ##                                           
    ##                Accuracy : 0.875           
    ##                  95% CI : (0.7873, 0.9359)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 2.377e-05       
    ##                                           
    ##                   Kappa : 0.7091          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.7857          
    ##             Specificity : 0.9167          
    ##          Pos Pred Value : 0.8148          
    ##          Neg Pred Value : 0.9016          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.2500          
    ##    Detection Prevalence : 0.3068          
    ##       Balanced Accuracy : 0.8512          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.lda <- rep("N", length(lda.pred))
test.lda[lda.pred > 0.4] <- "Y"
confusionMatrix(data = as.factor(test.lda), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 54 10
    ##          Y  6 18
    ##                                           
    ##                Accuracy : 0.8182          
    ##                  95% CI : (0.7216, 0.8924)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.003039        
    ##                                           
    ##                   Kappa : 0.5644          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.453255        
    ##                                           
    ##             Sensitivity : 0.6429          
    ##             Specificity : 0.9000          
    ##          Pos Pred Value : 0.7500          
    ##          Neg Pred Value : 0.8438          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.2045          
    ##    Detection Prevalence : 0.2727          
    ##       Balanced Accuracy : 0.7714          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.nb <- rep("N", length(nb.pred))
test.nb[glm.pred > 0.4] <- "Y"
confusionMatrix(data = as.factor(test.nb), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 54 11
    ##          Y  6 17
    ##                                           
    ##                Accuracy : 0.8068          
    ##                  95% CI : (0.7088, 0.8832)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.006377        
    ##                                           
    ##                   Kappa : 0.5325          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.331975        
    ##                                           
    ##             Sensitivity : 0.6071          
    ##             Specificity : 0.9000          
    ##          Pos Pred Value : 0.7391          
    ##          Neg Pred Value : 0.8308          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.1932          
    ##    Detection Prevalence : 0.2614          
    ##       Balanced Accuracy : 0.7536          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

# Conclusions

``` r
res <- resamples(list(GLM = glm.model,
                      GLMN = glmn.model,
                      GAM = gam.model,
                      MARS = mars.model,
                      LDA = lda.model,
                      NB = nb.model))
summary(res)
```

    ## 
    ## Call:
    ## summary.resamples(object = res)
    ## 
    ## Models: GLM, GLMN, GAM, MARS, LDA, NB 
    ## Number of resamples: 25 
    ## 
    ## ROC 
    ##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## GLM  0.7372449 0.8399015 0.8622449 0.8625304 0.9030612 0.9469496    0
    ## GLMN 0.7627551 0.8351648 0.8779841 0.8791417 0.9334975 0.9734748    0
    ## GAM  0.5782493 0.7576531 0.8078818 0.7931914 0.8594164 0.9581281    0
    ## MARS 0.6871921 0.8443878 0.8866995 0.8766179 0.9196429 0.9778325    0
    ## LDA  0.7525510 0.8214286 0.8596939 0.8618043 0.9124668 0.9469496    0
    ## NB   0.7315271 0.8341837 0.8719212 0.8657589 0.8928571 0.9753695    0
    ## 
    ## Sens 
    ##           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## GLM  0.7857143 0.8571429 0.8965517 0.8894089 0.9310345 1.0000000    0
    ## GLMN 0.9655172 1.0000000 1.0000000 0.9986207 1.0000000 1.0000000    0
    ## GAM  0.7241379 0.7931034 0.8275862 0.8476355 0.8965517 0.9642857    0
    ## MARS 0.7586207 0.8620690 0.8965517 0.9060099 0.9642857 0.9655172    0
    ## LDA  0.7500000 0.8620690 0.8965517 0.8837438 0.9285714 1.0000000    0
    ## NB   0.8620690 0.9310345 0.9642857 0.9553202 1.0000000 1.0000000    0
    ## 
    ## Spec 
    ##            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## GLM  0.28571429 0.5384615 0.6923077 0.6443956 0.7692308 0.9285714    0
    ## GLMN 0.00000000 0.0000000 0.0000000 0.0232967 0.0000000 0.1428571    0
    ## GAM  0.28571429 0.5384615 0.6428571 0.6237363 0.7142857 0.9285714    0
    ## MARS 0.28571429 0.6153846 0.6923077 0.6734066 0.7692308 1.0000000    0
    ## LDA  0.35714286 0.5384615 0.6923077 0.6531868 0.7857143 0.9285714    0
    ## NB   0.07142857 0.3571429 0.5000000 0.4503297 0.5714286 0.8461538    0

``` r
bwplot(res, metric = "ROC")
```

![](midterm_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

# Strength and Limitations

# Reference

1.Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA. Survival analysis of
heart failure patients: a case study. PLoS ONE. 2017; 12(7):0181001.
2.Zahid FM, Ramzan S, Faisal S, Hussain I. Gender based survival
prediction models for heart failure patients: a case study in Pakistan.
PLoS ONE. 2019; 14(2):0210602.
