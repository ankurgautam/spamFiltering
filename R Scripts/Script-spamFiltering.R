#################### Naive Bayes - Spam Filtering ############


# Problem Statement
# The input data is a set of SMS messages that has been classified as either “ham” or “spam”. The goal of the
# exercise is to build a model to identify messages as either ham or spam.



##################### Data Engineering & Analysis ##############################

setwd("E:/Mission Machine Learning/Git/spamFiltering")

sms_data <- read.csv("Data/sms_spam_short.csv", stringsAsFactors=FALSE)

sms_data$type <- as.factor(sms_data$type)

str(sms_data)

summary(sms_data)

head(sms_data)


# Data Cleansing
# The dataset contains raw text. The text need to be pre-processed and converted into a
# Document Term Matrix before it can be used for classification purposes. The steps required are documented
# as comments below

install.packages("tm")
library(tm)

#create a corpus for the message
mesg_corpus <- Corpus(VectorSource(sms_data$text))

#peek into the corpus
inspect(mesg_corpus[1:5])

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


#create a document-term sparse matrix
dtm <- DocumentTermMatrix(refined_corpus)
dtm

dim(dtm)

#Remove all words who has occured less than 10 times to create a new DTM
filtered_dtm <- DocumentTermMatrix(refined_corpus, list(dictionary=findFreqTerms(dtm, 10)))
dim(filtered_dtm)

#inspect the contents be converting it into a matrix and transposing it
t(inspect(filtered_dtm)[1:25,1:10])

# Exploratory Data Analysis - The following example shows a word cloud for both ham and spam message.
# The size of words shown in the word cloud is based on the frequency of occurance. It will clearly show that
# there is a difference in the most common occuring words between these types

install.packages("wordcloud")
library(wordcloud)

pal <- brewer.pal(9,"Dark2")

wordcloud(refined_corpus[sms_data$type=="ham"], min.freq=5,
          random.order=FALSE, colors=pal)


wordcloud(refined_corpus[sms_data$type=="spam"], min.freq=2,
          random.order=FALSE, colors=pal)





############################## Modeling & Prediction ###################################
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

#Model Building Build model based on the training data
library(e1071)

#Leave out the last column (target)
modFit <- naiveBayes(df_train[,-60], df_train$type)
modFit





#################################### Testing ##################################################
# Now let us predict the class for each sample in the test data. Then compare the prediction with the actual
# value of the class..

predictions <- predict(modFit, df_test)
confusionMatrix(predictions, df_test$type)

