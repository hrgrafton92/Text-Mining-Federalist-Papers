library(quanteda)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.textstats)
library(stopwords)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(dplyr)

papers <- read.csv("Data/federalist.csv",stringsAsFactors = F)

table(papers$Author)

papers <- papers[which(papers$Author == "HAMILTON" |
                         papers$Author == "MADISON" | 
                         papers$Author == "UNKNOWN"),]

papers <- papers[order(papers$Author),]

papers$Text <- substring(papers$Text,40)

myCorpus <- corpus(papers$Text)
summary(myCorpus)

myDfm <- dfm(tokens(myCorpus))
dim(myDfm)

tstat_freq <- textstat_frequency(myDfm)
head(tstat_freq,20)

library(ggplot2)
myDfm %>%
  textstat_frequency(n=20) %>%
  ggplot(aes(x=reorder(feature,frequency),y=frequency))+
  geom_point()+
  labs(x=NULL,y='frequency')+
  theme_minimal()

textplot_wordcloud(myDfm,max_words = 200)

library(stopwords)
myDfm <- dfm(tokens(myCorpus,
                    remove_punct = T))
myDfm <- dfm_remove(myDfm,stopwords('english'))
myDfm <- dfm_wordstem(myDfm)
dim(myDfm)

topfeatures(myDfm,30)

stopword1 <- c('may','one','two','can','must','upon','night','shall')
myDfm <- dfm_remove(myDfm,stopword1)
dim(myDfm)

topfeatures(myDfm,30)
textplot_wordcloud(myDfm)

stopwords2 <- c('state','govern','power','constitut','nation','peopl')
myDfm <- dfm_remove(myDfm,stopwords2)

myDfm <- dfm_trim(myDfm,min_termfreq = 4, min_docfreq = 2)
dim(myDfm)

myLda <- LDA(myDfm, k=8, control=list(seed=101))

myLda_td <- tidy(myLda)
myLda_td

top_terms <- myLda_td %>%
  group_by(topic) %>%
  top_n(8,beta) %>%
  ungroup() %>%
  arrange(topic,-beta)

top_terms %>%
  mutate(term=reorder(term,beta)) %>%
  ggplot(aes(term,beta,fill=factor(topic)))+
  geom_bar(stat='identity',show.legend=FALSE)+
  facet_wrap(~topic,scales='free')+
  coord_flip()

ap_documents <- tidy(myLda, matrix = 'gamma')
ap_documents

Lda_document <- as.data.frame(myLda@gamma)
Lda_document

modelDfm <- dfm(tokens(myCorpus,
                       remove_punct = T))
modelDfm <- dfm_remove(modelDfm,stopwords('english'))
modelDfm <- dfm_remove(modelDfm, stopword1)
modelDfm <- dfm_wordstem(modelDfm)

modelDfm <- dfm_trim(modelDfm, min_termfreq=4,min_docfreq = 2)
dim(modelDfm)

modelDfm_tfidf <- dfm_tfidf(modelDfm)
modelSvd <- textmodel_lsa(modelDfm_tfidf,nd=10)
head(modelSvd$docs)

modelData <- cbind(papers[,2],as.data.frame(modelSvd$docs))

colnames(modelData)[1] <- 'Author'
head(modelData)

trainData <- subset(modelData, Author=='HAMILTON'| Author == 'MADISON')
testData <- subset(modelData, Author == "UNKNOWN")

str(trainData)
trainData$Author <- factor(trainData$Author)

regModel <- glm(formula=Author~.,
                family=binomial(link=logit),
                data=trainData)

pred <- predict(regModel,newdata=trainData,type='response')
pred.result <- ifelse(pred > .5,1, 0)
print(table(pred.result,trainData$Author))

unknownPred <- predict(regModel,
                       newdata = testData,
                       type = 'response')
unknownPred <- cbind(testData$Author,as.data.frame(unknownPred))

unknownPred.result <- ifelse(pred > .5, 1,0)
