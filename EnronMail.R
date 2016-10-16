# 
# SENTIMENT ANALYSIS OF ENRON EMAILS 
# Working with Text
# Data file : energy_bids.csv
# Source : TREC Legal Track
# @author : Radhika Kini

library('RSentiment')   # Used to compute the sentiments of words
library('ggplot2')      # Plotting dynamic graphs
library('wordcloud')    # Getting a word cloud of frequency
library('tm')           # Used for pre-processing
library('SnowballC')    # Used for tm
library(caTools)        # used to split the dataset

#===========================================================================================
#                   Reading the text csv file into a data frame
#===========================================================================================
# Loading of the file into a dataframe and checing the features in order to apply a model

enronMailDF <- read.csv("energy_bids.csv", 
                        stringsAsFactors = FALSE) # strings shouldnt be stored as 
                                                  # factors to avoid discrepancy

str(enronMailDF)                                  # Check what are the contents of the datafile

# structure of enronMailDF
# 855 observations with 2 variables
# 1) email : contants of the email
# 2) responsive : is it responsive to our queries on energy bids 


strwrap(enronMailDF$email[[1]])                   # This wraps character strings into well defined paragraphs
                                                  # checks for the email in the data frame


table(enronMailDF$responsive)                     # the table is unbalanced value(0) >> value(1)


#============================================================================
#                                 Pre- processing
#============================================================================
# Creation of a corpus
# Corpus is a collection of documents

enronCorpus <- Corpus(VectorSource(enronMailDF$email))   # Converts into a corpus
enronCorpus

enronCorpus <-tm_map(enronCorpus, PlainTextDocument)     # Converts into plain text document

# the corpus contents
strwrap(enronCorpus[[1]])

# conversion of emails in lower case
enronCorpus <- tm_map(enronCorpus, tolower)
strwrap(enronCorpus[[1]])

# removal of  punctuation from the mails
enronCorpus <- tm_map(enronCorpus, removePunctuation)
strwrap(enronCorpus[[1]])

# Remove stop words
enronCorpus <- tm_map(enronCorpus, removeWords, c(stopwords(kind = "english")))
enronCorpus[[1]]

#stemming concept
enronCorpus <- tm_map(enronCorpus, stemDocument)
enronCorpus[[1]]

enronCorpus <- tm_map(enronCorpus, PlainTextDocument)
#Plot a word cloud of the frequent words
wordcloud(enronCorpus, max.words = 10,colors=brewer.pal(6,"Dark2"))

# Create a matrix of the corpus
enronMatrix <- DocumentTermMatrix(enronCorpus)
enronMatrix

# Remove the low frequency terms
enronMatrix <- removeSparseTerms(enronMatrix, 0.97)
enronMatrix

enronLabledTerms <- as.data.frame(as.matrix(enronMatrix))
str(enronLabledTerms)

enronLabledTerms$responsive = enronMailDF$responsive


str(enronLabledTerms)

# Compute the sentiment of each term in the data set
enronSentiment <- calculate_sentiment(names(enronLabledTerms))
enronSentiment
class(enronLabledTerms)

enronJustSentiment <- enronSentiment[,2]
enronJustSentiment

as.matrix(enronJustSentiment)
enronJustSentiment[788,] =c( "Neutral") 
for (i in 788:855){
  enronJustSentiment[[i]] = "Neutral"
}

enronJustSentiment
enronLabledTerms$Sentiment <-enronJustSentiment
str(enronLabledTerms)
enronLabledTerms$Sentiment
table(enronLabledTerms$Sentiment)
qplot(enronLabledTerms$Sentiment,
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2))

table(enronLabledTerms$Sentiment)

#========================================
# SPLIT data
#========================================

set.seed(144)     # Initializes the the random number generator
splitVector <- sample.split(enronLabledTerms$responsive, SplitRatio = 0.7) # Test and training are balanced data sets
splitVector

enronTrain <- subset(enronLabledTerms, splitVector==TRUE)
enronTest  <- subset(enronLabledTerms, splitVector== FALSE)

#===================================================================
# CART MODEL
#===================================================================

library(rpart)
library(rpart.plot)


enronCARTModel <- rpart(responsive ~.,      #Create a CART model enronCARTModel with method classification
                        data = enronTrain, 
                        method = "class")
prp(enronCARTModel, col = "dark blue")      # Display the model          


enronPredict <- predict(enronCARTModel,    # Predict the outcome of the testing set
                        enronTest, 
                        method = "class")
plot(enronPredict, col = "dark blue")
enronPredict[1:20, ]

enronPredict.Prob <- enronPredict[,2]     # The column 2 holds the responsiveness to be true
enronTest$responsive

# Accuracy
table(enronTest$responsive, enronPredict.Prob >=0.5)

accuracy <- (199+17)/(199+17+25+16)
accuracy

table(enronTest$responsive)
actualyAccuracy <- 215/(215+42)
actualyAccuracy

boxplot(accuracy, actualyAccuracy, col = "orange")

#============
# ROCR
#=============

library(ROCR)

predROCR <- prediction(enronPredict.Prob, enronTest$responsive)

perfROCR <- performance(predROCR, "tpr", "fpr")
plot(perfROCR, colorize = TRUE, print.cutoffs.at = seq(0,1,0.3), text.adj = c(-0.2, 0.7))

# AUC value = 83.57
perfAUC <- performance(predROCR, "auc")@y.values
perfAUC


