library(caret) # for using train function
library(randomForest) # for applying random forest
library(ggplot2)

# Read dataset
train_data <- read.csv("train.csv", header = TRUE)
test_data <- read.csv("test.csv", header = TRUE)

#============= Exploratory Analysis ======================
# quick summary of dataset
str(train_data)
str(test_data)

# Find the percentage of missing value in dataset
MissFun <- function(x){sum(is.na(x))/length(x)*100}
apply(train_data,2,MissFun) # 2 for column
apply(test_data, 2, MissFun)

# Fill the missing values with mean and median
train_data$Age[is.na(train_data$Age)] <- mean(train_data$Age, na.rm = TRUE)
test_data$Age[is.na(test_data$Age)] <- mean(test_data$Age, na.rm = TRUE)
test_data$Fare[is.na(test_data$Fare)] <- median(test_data$Fare, na.rm = TRUE)

# Convert Survived variable to factor from integer
train_data$Survived <- as.factor(train_data$Survived)

# Check the ratio of Survived by Pclass
table(train_data[,c("Survived", "Pclass")])

## Create PNG file
png(filename='SurClassPlot.png', width=480, height=480, units='px')

# Plot for the number of people Survived based on Pclass
ggplot(train_data, aes(Survived, Pclass, fill = Survived))+
  geom_bar(stat="identity")+ facet_grid(.~Pclass)+labs(x = "Survived/Dead", y="Counts")+
  ggtitle("Survived versus Pclass")

dev.off()

## Create PNG file
png(filename='SurAgePlot.png', width=480, height=480, units='px')

ggplot(train_data, aes(Survived, Age, fill = Survived))+geom_boxplot()+
  facet_grid(.~Pclass) + labs(x = "Survived/Dead", y="Age")+
  ggtitle("Survived versus Age Plot by Pclass")

dev.off()

#==================== Model with Random Forest =====================

set.seed(10)
# compare model1 and model2
model1 <- train(Survived ~ Pclass+Age+Sex+SibSp+Parch+Fare,
                data = train_data, method = "rf",
                trControl = trainControl(method = "cv", number = 5))

model2 <- train(Survived ~ Pclass+Age+Sex+SibSp+Parch+Fare+Embarked,
                data = train_data, method = "rf",
                trControl = trainControl(method = "cv", number = 5))
# model2 shows higher accuracy than model1

#=============== Accuracy of Model and Important Variables =====================
# Check accuracy of model2
pred1 <- predict(model2, train_data)
confusionMatrix(pred1, train_data$Survived)

# Check important variable from model2
impVar <- varImp(model2)
## Create PNG file
png(filename='impVarPlot.png', width=480, height=480, units='px')

plot(impVar, main = "Importance of Variables")

dev.off()

#============= Apply Model on Test Dataset========================

# Apply model2 in test_data
test_data$Survived <- predict(model2, test_data)

# select only the necessary variable needed for submission
submission <- test_data[,c("PassengerId", "Survived")]
write.table(submission, file = "submission_RandomForest.csv", col.names = TRUE,
            row.names = FALSE, sep = ",")