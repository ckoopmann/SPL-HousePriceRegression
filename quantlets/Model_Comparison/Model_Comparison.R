# clear variables and close windows
rm(list = ls(all = TRUE))
graphics.off()

# Install and load packages
libraries = c("ggplot2", "xtable", "gridExtra", "gbm", "plyr", "glmnet")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
      install.packages(x)
})

lapply(libraries, library, quietly = TRUE, character.only = TRUE)

#was machen wir hier eigentlich korrekterweise?
setwd("C:/Users/Felix/Uni/SPL/HousePriceRegression/quantlets/Model_Comparison")

# Read in pre-processed Data:
train = read.csv("train_preprocessed.csv")
test = read.csv("test_preprocessed.csv")

# Read in trained models
load("rf.RData")                    # random forest
load("gbm.RData")                   # gradient boosting model
load("regression_models_fit.RData") # OLS based models


#set rownumbers in dataframe to NULL
train$X  = NULL
train$Id = NULL
test$X   = NULL
test$Id  = NULL

# Comparing the models by mean squared error (MSE)
model.mse = function(model,test.data = test){
    if(class(model)[1] %in% c("train","lm")){
        pred = predict(model, newdata = test.data)
        mse  = (1/ncol(test.data))*sum((pred - test.data$logSalePrice)^2)
    }else{
        pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s ="lambda.1se")
        mse  = (1/ncol(test.data))*sum((pred - test.data$logSalePrice)^2)
    }
    return(mse)
}

model_overfit.mse = function(model,test.data = train){
      if(class(model)[1] %in% c("train","lm")){
            pred = predict(model, newdata = test.data)
            mse  = (1/ncol(test.data))*sum((pred - test.data$logSalePrice)^2)
      }else{
            pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s ="lambda.1se")
            mse  = (1/ncol(test.data))*sum((pred - test.data$logSalePrice)^2)
      }
      return(mse)
}


# Comparing the models by mean absolute errror
model.mae = function(model,test.data = test){
      if(class(model)[1] %in% c("train","lm")){
            pred = predict(model, newdata = test.data)
            mae  = (1/ncol(test.data))*sum(abs(pred - test.data$logSalePrice))
      }else{
            pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s ="lambda.1se")
            mae  = (1/ncol(test.data))*sum(abs(pred - test.data$logSalePrice))
      }
      return(mae)
}

# Comparing the models by bias
model.bias = function(model,test.data = test){
      if(class(model)[1] %in% c("train","lm")){
            pred = predict(model, newdata = test.data)
            bias  = (1/ncol(test.data))*sum(pred - test.data$logSalePrice)
      }else{
            pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s ="lambda.1se")
            bias  = (1/ncol(test.data))*sum(pred - test.data$logSalePrice)
      }
      return(bias)
}

# Comparing the models by R^2
model.Rsq = function(model,test.data = test){
      if(class(model)[1] %in% c("train","lm")){
            pred = predict(model, newdata = test.data)
            Rsq  = 1 - sum((pred - test.data$logSalePrice)^2)/sum((test.data$logSalePrice - mean(test.data$logSalePrice))^2)
      }else{
            pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s ="lambda.1se")
            Rsq  = 1 - sum((pred - test.data$logSalePrice)^2)/sum((test.data$logSalePrice - mean(test.data$logSalePrice))^2)
      }
      return(Rsq)
}

# comparing of the models using different measures
model.list              = list(lm.fit,fwd.fit,lasso.fit,ridge.fit,gbmtuned,rftuned)
comparisonMSE.list      = sapply(model.list, FUN = model.mse)
comparisonMSEtrain.list = sapply(model.list, FUN = model_overfit.mse)
comparisonMAE.list      = sapply(model.list, FUN = model.mae)
comparisonBIAS.list     = sapply(model.list, FUN = model.bias)
comparisonRSQ.list      = sapply(model.list, FUN = model.Rsq)
results                 = matrix(c(round(comparisonMSE.list,3),round(comparisonMSEtrain.list,3),round(comparisonMAE.list,3),round(comparisonBIAS.list,3),round(comparisonRSQ.list,3)),ncol = length(model.list), byrow = TRUE)
rownames(results)       = c("MSE","MSEtrain", "MAE","BIAS","RSQ")
colnames(results)       = c("lm","fwd","lasso","ridge","gbm","rf")
results

modelcomparison_latex = xtable(results)
print(modelcomparison_latex, file = "modelcomparison.tex")


# plot real values against estimated values
# lm.fit plot
df.lm.fit = data.frame(cbind(test$logSalePrice,predict(lm.fit, newdata = test)))
#pdf(file = "lm.fit_realVsfitted.pdf")
lm.plot = ggplot(df.lm.fit,aes(test$logSalePrice,predict(lm.fit, newdata = test))) + geom_point() + geom_segment(x = -4, y = -4,xend = 4, yend = 4, color = "red", size =1.3) +
    stat_smooth(method="lm", se=FALSE) +
    labs(title = "Plot of real logSalePrice against predicted values", x = "logSalePrice", y = "lm.fit predictions") + 
    theme(axis.title=element_text(size=16),  plot.title = element_text(size = 16, face = "bold")) +
    annotate("text", label = paste("MSE:",results["MSE","lm"], sep = " "), x = -3, y = 3) +
    annotate("text", label = paste("MAE:",results["MAE","lm"], sep = " "), x = -3, y = 2.5)
#dev.off()

# fwd.fit plot
df.fwd.fit = data.frame(cbind(test$logSalePrice,predict(fwd.fit, newdata = test)))
#pdf(file = "fwd.fit_realVsfitted.pdf")
fwd.plot = ggplot(df.fwd.fit,aes(test$logSalePrice,predict(fwd.fit, newdata = test))) + geom_point() + geom_segment(x = -4, y = -4,xend = 4, yend = 4, color = "red", size =1.3) +
    stat_smooth(method="lm", se=FALSE) +
    labs(title = "Plot of real logSalePrice against predicted values", x = "logSalePrice", y = "fwd.fit predictions") + 
    theme(axis.title=element_text(size=16),  plot.title = element_text(size = 16, face = "bold")) +
    annotate("text", label = paste("MSE:",results["MSE","fwd"], sep = " "), x = -3, y = 3) +
    annotate("text", label = paste("MAE:",results["MAE","fwd"], sep = " "), x = -3, y = 2.5)
#dev.off()

# lasso.fit plot
df.lasso.fit = data.frame(cbind(test$logSalePrice,predict(lasso.fit, newx = as.matrix(test[!names(test) %in% "logSalePrice"]), s ="lambda.1se")))
#pdf(file = "lasso.fit_realVsfitted.pdf")
lasso.plot = ggplot(df.lasso.fit,aes(test$logSalePrice,predict(lasso.fit, newx = as.matrix(test[!names(test) %in% "logSalePrice"]), s ="lambda.1se"))) + 
    geom_point() + geom_segment(x = -4, y = -4,xend = 4, yend = 4, color = "red", size =1.3) +
    stat_smooth(method="lm", se=FALSE) +
    labs(title = "Plot of real logSalePrice against predicted values", x = "logSalePrice", y = "lasso.fit predictions") + 
    theme(axis.title=element_text(size=16),  plot.title = element_text(size = 16, face = "bold")) +
    annotate("text", label = paste("MSE:",results["MSE","lasso"], sep = " "), x = -3, y = 3) +
    annotate("text", label = paste("MAE:",results["MAE","lasso"], sep = " "), x = -3, y = 2.5)
#dev.off()

# ridge.fit plot
df.ridge.fit = data.frame(cbind(test$logSalePrice,predict(ridge.fit, newx = as.matrix(test[!names(test) %in% "logSalePrice"]), s ="lambda.1se")))
#pdf(file = "ridge.fit_realVsfitted.pdf")
ridge.plot = ggplot(df.ridge.fit,aes(test$logSalePrice,predict(ridge.fit, newx = as.matrix(test[!names(test) %in% "logSalePrice"]), s ="lambda.1se"))) + 
    geom_point() + geom_segment(x = -4, y = -4,xend = 4, yend = 4, color = "red", size =1.3) +
    stat_smooth(method="lm", se=FALSE) +
    labs(title = "Plot of real logSalePrice against predicted values", x = "logSalePrice", y = "ridge.fit predictions") + 
    theme(axis.title=element_text(size=16),  plot.title = element_text(size = 16, face = "bold")) +
    annotate("text", label = paste("MSE:",results["MSE","ridge"], sep = " "), x = -3, y = 3) +
    annotate("text", label = paste("MAE:",results["MAE","ridge"], sep = " "), x = -3, y = 2.5)
#dev.off()

# gbm plot
#prop.overestimated.gbm = (sum(predict(gbmtuned, newdata = test)-test$logSalePrice > 0)/length(test$logSalePrice))*100
df.gbm = data.frame(cbind(test$logSalePrice,predict(gbmtuned, newdata = test)))
#pdf(file = "gbm_realVsfitted.pdf")
gbm.plot = ggplot(df.gbm,aes(test$logSalePrice,predict(gbmtuned, newdata = test))) + geom_point() + geom_segment(x = -4, y = -4,xend = 4, yend = 4, color = "red", size =1.3) +
    stat_smooth(method="lm", se=FALSE) +
    labs(title = "Plot of real logSalePrice against predicted values", x = "logSalePrice", y = "gbmtuned predictions") + 
    theme(axis.title=element_text(size=16),  plot.title = element_text(size = 16, face = "bold")) +
    annotate("text", label = paste("MSE:",results["MSE","gbm"], sep = " "), x = -3, y = 3) +
    annotate("text", label = paste("MAE:",results["MAE","gbm"], sep = " "), x = -3, y = 2.5)
#dev.off()

# rf plot
#prop.overestimated.rf = (sum(predict(rftuned, newdata = test)-test$logSalePrice > 0)/length(test$logSalePrice))*100
df.rf = data.frame(cbind(test$logSalePrice,predict(rftuned, newdata = test)))
#pdf(file = "rf_realVsfitted.pdf")
rf.plot = ggplot(df.rf,aes(test$logSalePrice,predict(rftuned, newdata = test))) + geom_point() + geom_segment(x = -4, y = -4,xend = 4, yend = 4, color = "red", size =1.3) +
    stat_smooth(method="lm", se=FALSE) +
    labs(title = "Plot of real logSalePrice against predicted values", x = "logSalePrice", y = "rftuned predictions") + 
    theme(axis.title=element_text(size=16),  plot.title = element_text(size = 16, face = "bold")) +
    annotate("text", label = paste("MSE:",results["MSE","rf"], sep = " "), x = -3, y = 3) +
    annotate("text", label = paste("MAE:",results["MAE","rf"], sep = " "), x = -3, y = 2.5)
#dev.off()

# plotting resulting graphs together
source("http://peterhaschke.com/Code/multiplot.R")
pdf(file = "Model_Comparison.pdf")
multiplot(lm.plot, fwd.plot, lasso.plot, ridge.plot, gbm.plot, ridge.plot, cols=2)
dev.off()