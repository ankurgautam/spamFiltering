# Naive Bayes - Spam Filtering

## Problem Statement
The input data is a set of SMS messages that has been classified as either “ham” or “spam”. The goal of the
exercise is to build a model to identify messages as either ham or spam.

##Techniques Used
1. Naive Bayes Classifier
2. Training and Testing
3. Confusion Matrix
4. Text Pre-Processing

##Data Engineering & Analysis


###Loading and understanding the dataset
```{r}
setwd("E:/Mission Machine Learning/Git/spamFiltering")

sms_data <- read.csv("sms_spam_short.csv", stringsAsFactors=FALSE)

sms_data$type <- as.factor(sms_data$type)

str(sms_data)
```

```
## 'data.frame': 500 obs. of 2 variables:
## $ type: Factor w/ 2 levels "ham","spam": 1 1 1 2 2 1 1 1 2 1 ...
## $ text: chr "Hope you are having a good week. Just checking in" "K..give back my thanks." "Am also
```
```{r}
summary(sms_data)
```
```
## type text
## ham :437 Length:500
## spam: 63 Class :character
## Mode :character
```
```{r}
head(sms_data)
```
```
## type
## 1 ham
## 2 ham
## 3 ham
## 4 spam
## 5 spam
## 6 ham
## 
## 1 
## 2 
## 3 
## 4 complimentary 4 STAR Ibiza Holiday or Â£10,000 cash needs your URGENT collection. 09066364349 ## 5 okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award! 
## 6
```

Data Cleansing The dataset contains raw text. The text need to be pre-processed and converted into a
Document Term Matrix before it can be used for classification purposes. The steps required are documented
as comments below



```{r}
library(tm)

#create a corpus for the message
mesg_corpus <- Corpus(VectorSource(sms_data$text))

#peek into the corpus
inspect(mesg_corpus[1:5])

```


```
## <<VCorpus (documents: 5, metadata (corpus/indexed): 0/0)>>
##
## [[1]]
## <<PlainTextDocument (metadata: 7)>>
## Hope you are having a good week. Just checking in
##
## [[2]]
2
## <<PlainTextDocument (metadata: 7)>>
## K..give back my thanks.
##
## [[3]]
## <<PlainTextDocument (metadata: 7)>>
## Am also doing in cbe only. But have to pay.
##
## [[4]]
## <<PlainTextDocument (metadata: 7)>>
## complimentary 4 STAR Ibiza Holiday or Â£10,000 cash needs your URGENT collection. 09066364349 NOW from ##
## [[5]]
## <<PlainTextDocument (metadata: 7)>>
## okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award!
```

```{r}
#cleanse the data
#remove punctuation marks
refined_corpus <- tm_map(mesg_corpus, removePunctuation)

#remove white space
refined_corpus <- tm_map(refined_corpus, stripWhitespace)

#convert to lower case
refined_corpus <- tm_map(refined_corpus, content_transformer(tolower))

#remove numbers in text
refined_corpus <- tm_map(refined_corpus, removeNumbers)

#remove stop words
refined_corpus <- tm_map(refined_corpus, removeWords, stopwords())

#remove specific words
refined_corpus <- tm_map(refined_corpus, removeWords, c("else","the","are","for",
"has","they","as","a","his","on","when","is","in","already"))

#look at the processed text
inspect(refined_corpus[1:5])
```
```
## <<VCorpus (documents: 5, metadata (corpus/indexed): 0/0)>>
##
## [[1]]
## <<PlainTextDocument (metadata: 7)>>
## hope good week just checking
##
## [[2]]
## <<PlainTextDocument (metadata: 7)>>
## kgive back thanks
##
## [[3]]
## <<PlainTextDocument (metadata: 7)>>
## also cbe pay
##
## [[4]]
## <<PlainTextDocument (metadata: 7)>>
## complimentary star ibiza holiday â cash needs urgent collection now landline lose boxskwpppm
##
## [[5]]
## <<PlainTextDocument (metadata: 7)>>
## okmail dear dave final notice collect tenerife holiday cash award call landline tcs sae b
```


```{r}
#create a document-term sparse matrix
dtm <- DocumentTermMatrix(refined_corpus)
dtm
```

```
## <<DocumentTermMatrix (documents: 500, terms: 1966)>>
## Non-/sparse entries: 4021/978979
## Sparsity : 100%
## Maximal term length: 33
## Weighting : term frequency (tf)
```

```{r}
#Remove all words who has occured less than 10 times to create a new DTM
filtered_dtm <- DocumentTermMatrix(refined_corpus, list(dictionary=findFreqTerms(dtm, 10)))
dim(filtered_dtm)

#inspect the contents be converting it into a matrix and transposing it
t(inspect(filtered_dtm)[1:25,1:10])
```
```
## <<DocumentTermMatrix (documents: 500, terms: 59)>>
## Non-/sparse entries: 943/28557
## Sparsity : 97%
## Maximal term length: 8
## Weighting : term frequency (tf)
##
## Terms
## Docs anything back box call can care claim come day didnt dont free get
## 1 0 0 0 0 0 0 0 0 0 0 0 0 0
## 2 0 1 0 0 0 0 0 0 0 0 0 0 0
## 3 0 0 0 0 0 0 0 0 0 0 0 0 0
## 4 0 0 0 0 0 0 0 0 0 0 0 0 0
## 5 0 0 1 1 0 0 0 0 0 0 0 0 0
## 6 0 0 0 0 0 0 0 0 0 0 0 0 0
```

Exploratory Data Analysis - The following example shows a word cloud for both ham and spam message.
The size of words shown in the word cloud is based on the frequency of occurance. It will clearly show that
there is a difference in the most common occuring words between these types


```{r}
library(wordcloud)

pal <- brewer.pal(9,"Dark2")

wordcloud(refined_corpus[sms_data$type=="ham"], min.freq=5,
random.order=FALSE, colors=pal)
```

Word Cloud:
![alt text](https://github.com/ankurgautam/spamFiltering/blob/master/Viz/wordcloud1.png "Word Cloud")


```{r}
wordcloud(refined_corpus[sms_data$type=="spam"], min.freq=2,
random.order=FALSE, colors=pal)
```

Word Cloud:
![alt text](https://github.com/ankurgautam/spamFiltering/blob/master/Viz/wordcloud2.png "Word Cloud")

##Modeling & Prediction
```{r}
library(caret)

inTrain <- createDataPartition(y=sms_data$type ,p=0.7,list=FALSE)

#Spliting the raw data
train_raw <- sms_data[inTrain,]
test_raw <- sms_data[-inTrain,]

#spliting the corpus
train_corpus <- refined_corpus[inTrain]
test_corpus <- refined_corpus[-inTrain]

#spliting the dtm
train_dtm <- filtered_dtm[inTrain,]
test_dtm <-filtered_dtm[-inTrain,]

# Instead of using the counts of words within document, we will replace them with indicators "Yes" or "No".
# Yes indicates if the word occured in the document and No indicate it does not. This procedure converts
# Numeric data into factor data
conv_counts <- function(x) {
x <- ifelse(x > 0, 1, 0)
x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}
train <- apply(train_dtm, MARGIN = 2, conv_counts)
test <- apply(test_dtm, MARGIN = 2, conv_counts)

#convert to a data frame and add the target variable
df_train <- as.data.frame(train)
df_test <- as.data.frame(test)
df_train$type <- train_raw$type
df_test$type <- test_raw$type
df_train[1:10,1:10]
```
```
## anything back box call can care claim come day didnt
## 1 No No No No No No No No No No
## 10 No No No No No No No No No No
## 11 No No No No No No No No No No
## 12 Yes No No No No No No No No No
## 13 No No No Yes No No No No No No
## 14 No No No No No No No No No No
## 18 Yes No No No No No No No No No
## 21 No No No No No No No No No No
## 22 No No No No No No No No No No
## 23 No No No No No No No No No No
```

```{r}
#Model Building - Build model based on the training data
library(e1071)
```

```
#Leave out the last column (target)
modFit <- naiveBayes(df_train[,-60], df_train$type)
modFit
```


##Testing

Predict the class for each sample in the test data. Then compare the prediction with the actual
value of the class

```{r}
predictions <- predict(modFit, df_test)
confusionMatrix(predictions, df_test$type)
```
```
## Confusion Matrix and Statistics
##
## Reference
60
## Prediction ham spam
## ham 126 3
## spam 5 15
##
## Accuracy : 0.946
## 95% CI : (0.897, 0.977)
## No Information Rate : 0.879
## P-Value [Acc > NIR] : 0.00476
##
## Kappa : 0.759
## Mcnemar's Test P-Value : 0.72367
##
## Sensitivity : 0.962
## Specificity : 0.833
## Pos Pred Value : 0.977
## Neg Pred Value : 0.750
## Prevalence : 0.879
## Detection Rate : 0.846
## Detection Prevalence : 0.866
## Balanced Accuracy : 0.898
##
## 'Positive' Class : ham
```


