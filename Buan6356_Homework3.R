library(dplyr)
library(lattice)
library(ggplot2)
library(caret)
library(MASS)
library(tidyverse)


f <- file("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", open="r" ,encoding="UTF-8")
data <- read.table(f, sep=',')

names(data) <- c("Freq_of_make", 
                 "Freq_of_address", 
                 "Freq_of_all", 
                 "Freq_of_3d", 
                 "Freq_of_our", 
                 "Freq_of_over", 
                 "Freq_of_remove", 
                 "Freq_of_internet", 
                 "Freq_of_order", 
                 "Freq_of_mail", 
                 "Freq_of_receive", 
                 "Freq_of_will", 
                 "Freq_of_people", 
                 "Freq_of_report", 
                 "Freq_of_addresses", 
                 "Freq_of_free", 
                 "Freq_of_business", 
                 "Freq_of_email", 
                 "Freq_of_you", 
                 "Freq_of_credit", 
                 "Freq_of_your", 
                 "Freq_of_font", 
                 "Freq_of_000", 
                 "Freq_of_money", 
                 "Freq_of_hp", 
                 "Freq_of_hpl", 
                 "Freq_of_george", 
                 "Freq_of_650", 
                 "Freq_of_lab", 
                 "Freq_of_labs", 
                 "Freq_of_telnet", 
                 "Freq_of_857", 
                 "Freq_of_data", 
                 "Freq_of_415", 
                 "Freq_of_85", 
                 "Freq_of_technology", 
                 "Freq_of_1999", 
                 "Freq_of_parts", 
                 "Freq_of_pm", 
                 "Freq_of_direct", 
                 "Freq_of_cs", 
                 "Freq_of_meeting", 
                 "Freq_of_original", 
                 "Freq_of_project", 
                 "Freq_of_re", 
                 "Freq_of_edu", 
                 "Freq_of_table", 
                 "Freq_of_conference", 
                 "Freq_of_semi-colon", 
                 "Freq_of_opening_bracket", 
                 "Freq_of_opening_bracebracket", 
                 "Freq_of_exclamation_mark", 
                 "Freq_of_dollar_sign", 
                 "Freq_of_pound_sign", 
                 "average_CapitalLetters_length", 
                 "longest_CapitalLetters_length", 
                 "Total_CapitalLetters_length", 
                 "Spam"
)

options(scipen = 999)

# Preprocess Data
Data.df <- preProcess(data[,-58], method = c("center", "scale"))
scaled.df <- predict(Data.df, data)
spam.df <- scaled.df[which(scaled.df$Spam==1),]
Meanspam.df <- colMeans(spam.df[,-58])
nonspam.df <- scaled.df[which(scaled.df$Spam==0),]
Meannonspam.df <- colMeans(nonspam.df[,-58])
Difference <- abs(Meannonspam.df - Meanspam.df)
SortedDifference <- sort(Difference,decreasing = T)

# 10 main predictors
names(head(SortedDifference, n=10))

# Create Paritions
set.seed(123)
train.index <- createDataPartition(scaled.df$Spam, p = 0.8, list = FALSE)
train.df <- scaled.df[train.index, ]
valid.df <- scaled.df[-train.index, ]

# Run LDA
lda.train <- lda(Spam ~ Freq_of_0 + Freq_of_you + Freq_of_your + Freq_of_free + Freq_of_our + Freq_of_remove + 
                   Total_CapitalLetters_length + Freq_of_business + Freq_of_hp + Freq_of_dollar_sign, data = train.df)


# Checking accuracy of the model. We are getting a near 84% accuracy, which is good
SpamValidationPred <- predict(lda.train, valid.df)
confusionMatrix(as.factor(SpamValidationPred$class), as.factor(valid.df$Spam))

# Calculate Gains
gain <- gains(as.numeric(valid.df$Spam), SpamValidationPred$posterior[,2], groups = 10)

### Plot Lift Chart
# spam  <- as.numeric(spamdata.df.valid$classifier_actual)
plot(c(0,gain$cume.pct.of.total*sum(valid.df$Spam))~c(0,gain$cume.obs), 
     xlab = "Number of Cases", ylab = "Cumulative", main = "", type = "l")
lines(c(0,sum(valid.df$Spam))~c(0, dim(valid.df)[1]), lty = 5)

# Plot Decile-wise chart
heights <- gain$mean.resp/mean(valid.df$Spam)
midpoints <- barplot(heights, names.arg = gain$depth,  ylim = c(0,9), col = "red",
                     xlab = "Percentile", ylab = "Mean Response",
                     main = "Decile-wise Lift Chart")
# Add Labels to Columns
text(midpoints, heights+0.5, labels=round(heights, 1), cex = 0.8)
