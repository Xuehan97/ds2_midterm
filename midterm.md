Report
================
Xuehan Yang
2022/3/24

``` r
library(tidyverse)
library(caret)
library(pROC)
library(pdp)
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

![](midterm_files/figure-gfm/unnamed-chunk-3-1.png)<!-- --> Among all
the continuous predictors, we can see that death events grouped when the
follow-up-period is short(time &lt; 50), patients with higher level of
serum sodium tended to have death event, and patients with lower
ejection fraction tended to have death event.

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
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
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

    ##     alpha    lambda
    ## 179   0.2 0.1704641

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
    ## 1.5452 2.1921 5.6651 1.4792 8.9997 0.0000 0.0005 
    ##  total = 25.88 
    ## 
    ## UBRE score: -0.2829606

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
    ## (Intercept)         -1.97924    1.10140  -1.797   0.0723 .
    ## anaemia             -1.13985    0.62895  -1.812   0.0699 .
    ## diabetes            -0.27436    0.58195  -0.471   0.6373  
    ## high_blood_pressure -0.09577    0.59653  -0.161   0.8725  
    ## sex                 -1.01700    0.66772  -1.523   0.1277  
    ## smoking             -0.20193    0.64838  -0.311   0.7555  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Approximate significance of smooth terms:
    ##                                   edf Ref.df Chi.sq  p-value    
    ## s(ejection_fraction)        1.545e+00      9 13.154 0.000337 ***
    ## s(serum_sodium)             2.192e+00      9  5.330 0.059921 .  
    ## s(serum_creatinine)         5.665e+00      9 10.129 0.083384 .  
    ## s(age)                      1.479e+00      9  6.432 0.012822 *  
    ## s(time)                     9.000e+00      9 34.476 4.85e-05 ***
    ## s(platelets)                1.115e-05      9  0.000 0.455619    
    ## s(creatinine_phosphokinase) 5.186e-04      9  0.000 0.319798    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## R-sq.(adj) =  0.648   Deviance explained = 62.5%
    ## UBRE = -0.28296  Scale est. = 1         n = 211

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
    ##             glm=list(family=function.object, maxit=100), degree=1, nprune=6)
    ## 
    ## GLM coefficients
    ##                                    Y
    ## (Intercept)              -1.01927663
    ## h(age-67)                 0.11401409
    ## h(35-ejection_fraction)   0.18385539
    ## h(1.83-serum_creatinine) -1.79626411
    ## h(80-time)                0.05326049
    ## h(time-80)               -0.00856679
    ## 
    ## GLM (family binomial, link logit):
    ##  nulldev  df       dev  df   devratio     AIC iters converged
    ##  265.257 210   135.406 205       0.49   147.4     6         1
    ## 
    ## Earth selected 6 of 19 terms, and 4 of 12 predictors (nprune=6)
    ## Termination condition: Reached nk 25
    ## Importance: time, ejection_fraction, serum_creatinine, age, anaemia-unused, ...
    ## Number of terms at each degree of interaction: 1 5 (additive model)
    ## Earth GCV 0.1162873    RSS 22.04499    GRSq 0.472617    RSq 0.5216481

``` r
mars.model$bestTune
```

    ##   nprune degree
    ## 5      6      1

``` r
pdp::partial(mars.model, pred.var = c("age"), grid.resolution = 200) %>% autoplot()
```

    ## Warning: Use of `object[[1L]]` is discouraged. Use `.data[[1L]]` instead.

    ## Warning: Use of `object[["yhat"]]` is discouraged. Use `.data[["yhat"]]`
    ## instead.

![](midterm_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

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

![](midterm_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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
test.glm[glm.pred > 0.5] <- "Y"
confusionMatrix(data = as.factor(test.glm), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 55 12
    ##          Y  5 16
    ##                                           
    ##                Accuracy : 0.8068          
    ##                  95% CI : (0.7088, 0.8832)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.006377        
    ##                                           
    ##                   Kappa : 0.523           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.145610        
    ##                                           
    ##             Sensitivity : 0.5714          
    ##             Specificity : 0.9167          
    ##          Pos Pred Value : 0.7619          
    ##          Neg Pred Value : 0.8209          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.1818          
    ##    Detection Prevalence : 0.2386          
    ##       Balanced Accuracy : 0.7440          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.glmn <- rep("N", length(glmn.pred))
test.glmn[glmn.pred > 0.5] <- "Y"
confusionMatrix(data = as.factor(test.glmn), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 60 20
    ##          Y  0  8
    ##                                           
    ##                Accuracy : 0.7727          
    ##                  95% CI : (0.6711, 0.8553)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.0401          
    ##                                           
    ##                   Kappa : 0.3529          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.152e-05       
    ##                                           
    ##             Sensitivity : 0.28571         
    ##             Specificity : 1.00000         
    ##          Pos Pred Value : 1.00000         
    ##          Neg Pred Value : 0.75000         
    ##              Prevalence : 0.31818         
    ##          Detection Rate : 0.09091         
    ##    Detection Prevalence : 0.09091         
    ##       Balanced Accuracy : 0.64286         
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.gam <- rep("N", length(gam.pred))
test.gam[gam.pred > 0.5] <- "Y"
confusionMatrix(data = as.factor(test.gam), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 56 15
    ##          Y  4 13
    ##                                           
    ##                Accuracy : 0.7841          
    ##                  95% CI : (0.6835, 0.8647)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.02309         
    ##                                           
    ##                   Kappa : 0.4441          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.02178         
    ##                                           
    ##             Sensitivity : 0.4643          
    ##             Specificity : 0.9333          
    ##          Pos Pred Value : 0.7647          
    ##          Neg Pred Value : 0.7887          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.1477          
    ##    Detection Prevalence : 0.1932          
    ##       Balanced Accuracy : 0.6988          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.mars <- rep("N", length(mars.pred))
test.mars[mars.pred > 0.5] <- "Y"
confusionMatrix(data = as.factor(test.mars), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 58  8
    ##          Y  2 20
    ##                                           
    ##                Accuracy : 0.8864          
    ##                  95% CI : (0.8009, 0.9441)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 6.9e-06         
    ##                                           
    ##                   Kappa : 0.7222          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.1138          
    ##                                           
    ##             Sensitivity : 0.7143          
    ##             Specificity : 0.9667          
    ##          Pos Pred Value : 0.9091          
    ##          Neg Pred Value : 0.8788          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.2273          
    ##    Detection Prevalence : 0.2500          
    ##       Balanced Accuracy : 0.8405          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.lda <- rep("N", length(lda.pred))
test.lda[lda.pred > 0.5] <- "Y"
confusionMatrix(data = as.factor(test.lda), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 56 11
    ##          Y  4 17
    ##                                           
    ##                Accuracy : 0.8295          
    ##                  95% CI : (0.7345, 0.9013)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.00135         
    ##                                           
    ##                   Kappa : 0.5791          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.12134         
    ##                                           
    ##             Sensitivity : 0.6071          
    ##             Specificity : 0.9333          
    ##          Pos Pred Value : 0.8095          
    ##          Neg Pred Value : 0.8358          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.1932          
    ##    Detection Prevalence : 0.2386          
    ##       Balanced Accuracy : 0.7702          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

``` r
test.nb <- rep("N", length(nb.pred))
test.nb[glm.pred > 0.5] <- "Y"
confusionMatrix(data = as.factor(test.nb), reference = hf_df$death_event[-trainrows], positive = "Y")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  N  Y
    ##          N 55 12
    ##          Y  5 16
    ##                                           
    ##                Accuracy : 0.8068          
    ##                  95% CI : (0.7088, 0.8832)
    ##     No Information Rate : 0.6818          
    ##     P-Value [Acc > NIR] : 0.006377        
    ##                                           
    ##                   Kappa : 0.523           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.145610        
    ##                                           
    ##             Sensitivity : 0.5714          
    ##             Specificity : 0.9167          
    ##          Pos Pred Value : 0.7619          
    ##          Neg Pred Value : 0.8209          
    ##              Prevalence : 0.3182          
    ##          Detection Rate : 0.1818          
    ##    Detection Prevalence : 0.2386          
    ##       Balanced Accuracy : 0.7440          
    ##                                           
    ##        'Positive' Class : Y               
    ## 

# Conclusions

# Strength and Limitations

# Reference

1.Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA. Survival analysis of
heart failure patients: a case study. PLoS ONE. 2017; 12(7):0181001.
2.Zahid FM, Ramzan S, Faisal S, Hussain I. Gender based survival
prediction models for heart failure patients: a case study in Pakistan.
PLoS ONE. 2019; 14(2):0210602.
