# clear variables and close windows
rm(list = ls(all = TRUE))
graphics.off()

# Install and load packages
libraries = c("ggplot2", "xtable", "gridExtra", "gbm", "plyr", "glmnet")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
    install.packages(x)
})

lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# setting up working directory
setwd("C:/Users/Felix/Uni/SPL/HousePriceRegression/quantlets/Model_Comparison")

# Read in pre-processed Data:
train = read.csv("train_preprocessed.csv")
test  = read.csv("test_preprocessed.csv")

# Read in trained models
load("rf.RData")  # random forest
load("gbm.RData")  # gradient boosting model
load("regression_models_fit.RData")  # OLS based models


# set rownumbers in dataframe to NULL
train$X  = NULL
train$Id = NULL
test$X   = NULL
test$Id  = NULL

# Comparing the models by mean squared error (MSE)
model.mse = function(model, test.data = test) {
    if (class(model)[1] %in% c("train", "lm")) {
        pred = predict(model, newdata = test.data)
        mse  = (1/ncol(test.data)) * sum((pred - test.data$logSalePrice)^2)
    } else {
        pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s = "lambda.1se")
        mse  = (1/ncol(test.data)) * sum((pred - test.data$logSalePrice)^2)
    }
    return(mse)
}

# Comparing the models by MSE on the training data
model_overfit.mse = function(model, test.data = train) {
    if (class(model)[1] %in% c("train", "lm")) {
        pred = predict(model, newdata = test.data)
        mse  = (1/ncol(test.data)) * sum((pred - test.data$logSalePrice)^2)
    } else {
        pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s = "lambda.1se")
        mse  = (1/ncol(test.data)) * sum((pred - test.data$logSalePrice)^2)
    }
    return(mse)
}


# Comparing the models by mean absolute errror
model.mae = function(model, test.data = test) {
    if (class(model)[1] %in% c("train", "lm")) {
        pred = predict(model, newdata = test.data)
        mae  = (1/ncol(test.data)) * sum(abs(pred - test.data$logSalePrice))
    } else {
        pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s = "lambda.1se")
        mae  = (1/ncol(test.data)) * sum(abs(pred - test.data$logSalePrice))
    }
    return(mae)
}

# Comparing the models by bias
model.bias = function(model, test.data = test) {
    if (class(model)[1] %in% c("train", "lm")) {
        pred = predict(model, newdata = test.data)
        bias = (1/ncol(test.data)) * sum(pred - test.data$logSalePrice)
    } else {
        pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s = "lambda.1se")
        bias = (1/ncol(test.data)) * sum(pred - test.data$logSalePrice)
    }
    return(bias)
}

# Comparing the models by R^2
model.Rsq = function(model, test.data = test) {
    if (class(model)[1] %in% c("train", "lm")) {
        pred = predict(model, newdata = test.data)
        Rsq  = 1 - sum((test.data$logSalePrice - pred)^2)/sum((test.data$logSalePrice - mean(test.data$logSalePrice))^2)
    } else {
        pred = predict(model, newx = as.matrix(test.data[!names(test.data) %in% "logSalePrice"]), s = "lambda.1se")
        Rsq  = 1 - sum((test.data$logSalePrice - pred)^2)/sum((test.data$logSalePrice - mean(test.data$logSalePrice))^2)
    }
    return(Rsq)
}

# comparing of the models using different measures
model.list                  = list(lm.fit, fwd.fit, lasso.fit, ridge.fit, gbmtuned, rftuned)
comparisonMSE.list          = sapply(model.list, FUN = model.mse)
comparisonMSEtrain.list     = sapply(model.list, FUN = model_overfit.mse)
comparisonMAE.list          = sapply(model.list, FUN = model.mae)
comparisonBIAS.list         = sapply(model.list, FUN = model.bias)
comparisonRSQ.list          = sapply(model.list, FUN = model.Rsq)
comparison.result           = matrix(c(round(comparisonMSE.list, 3), round(comparisonMSEtrain.list, 3), round(comparisonMAE.list, 3), 
    round(comparisonBIAS.list, 3), round(comparisonRSQ.list, 3)), ncol = length(model.list), byrow = TRUE)
rownames(comparison.result) = c("MSE", "MSEtrain", "MAE", "BIAS", "RSQ")
colnames(comparison.result) = c("lm", "fwd", "lasso", "ridge", "gbm", "rf")
comparison.result

# Writing a latex table containing the model comparison results
modelcomparison_latex = xtable(comparison.result)
print(modelcomparison_latex, file = "modelcomparison.tex")

#################################################################################
# Plotting the estimations against the real values in the test dataset


# plot real values against estimated values lm.fit plot
predictions.lm = predict(lm.fit, newdata = test)                        # predicting the outcome for test data
df.lm.fit      = data.frame(cbind(test$logSalePrice, predictions.lm ))  # creating dataframe containing real and predicted outcome

lm.plot = ggplot(df.lm.fit, aes(test$logSalePrice, predictions.lm)) + geom_point() + geom_segment(x = -4, 
    y = -4, xend = 4, yend = 4, color = "red", size = 1.3) + stat_smooth(method = "lm", se = FALSE) + labs(title = "Plot of real logSalePrice against predicted values", 
    x = "logSalePrice", y = "lm.fit predictions") + theme(axis.title = element_text(size = 16), plot.title = element_text(size = 16, 
    face = "bold")) + annotate("text", label = paste("MSE:", comparison.result["MSE", "lm"], sep = " "), x = -3, y = 3) + annotate("text", 
    label = paste("MAE:", comparison.result["MAE", "lm"], sep = " "), x = -3, y = 2.5)


# fwd.fit plot
predictions.fwd = predict(fwd.fit, newdata = test)
df.fwd.fit      = data.frame(cbind(test$logSalePrice, predictions.fwd))

fwd.plot = ggplot(df.fwd.fit, aes(test$logSalePrice, predictions.fwd)) + geom_point() + geom_segment(x = -4, 
    y = -4, xend = 4, yend = 4, color = "red", size = 1.3) + stat_smooth(method = "lm", se = FALSE) + labs(title = "Plot of real logSalePrice against predicted values", 
    x = "logSalePrice", y = "fwd.fit predictions") + theme(axis.title = element_text(size = 16), plot.title = element_text(size = 16, 
    face = "bold")) + annotate("text", label = paste("MSE:", comparison.result["MSE", "fwd"], sep = " "), x = -3, y = 3) + 
    annotate("text", label = paste("MAE:", comparison.result["MAE", "fwd"], sep = " "), x = -3, y = 2.5)


# lasso.fit plot
predictions.lasso = predict(lasso.fit, newx = as.matrix(test[!names(test) %in% "logSalePrice"]), s = "lambda.1se")
df.lasso.fit      = data.frame(cbind(test$logSalePrice, predictions.lasso))

lasso.plot = ggplot(df.lasso.fit, aes(test$logSalePrice, predictions.lasso)) + geom_point() + geom_segment(x = -4, y = -4, xend = 4, yend = 4, color = "red", 
    size = 1.3) + stat_smooth(method = "lm", se = FALSE) + labs(title = "Plot of real logSalePrice against predicted values", 
    x = "logSalePrice", y = "lasso.fit predictions") + theme(axis.title = element_text(size = 16), plot.title = element_text(size = 16, 
    face = "bold")) + annotate("text", label = paste("MSE:", comparison.result["MSE", "lasso"], sep = " "), x = -3, y = 3) + 
    annotate("text", label = paste("MAE:", comparison.result["MAE", "lasso"], sep = " "), x = -3, y = 2.5)


# ridge.fit plot
predicions.ridge = predict(ridge.fit, newx = as.matrix(test[!names(test) %in% "logSalePrice"]), s = "lambda.1se")
df.ridge.fit     = data.frame(cbind(test$logSalePrice, predicions.ridge))

ridge.plot = ggplot(df.ridge.fit, aes(test$logSalePrice, predicions.ridge)) + geom_point() + geom_segment(x = -4, y = -4, xend = 4, yend = 4, color = "red", 
    size = 1.3) + stat_smooth(method = "lm", se = FALSE) + labs(title = "Plot of real logSalePrice against predicted values", 
    x = "logSalePrice", y = "ridge.fit predictions") + theme(axis.title = element_text(size = 16), plot.title = element_text(size = 16, 
    face = "bold")) + annotate("text", label = paste("MSE:", comparison.result["MSE", "ridge"], sep = " "), x = -3, y = 3) + 
    annotate("text", label = paste("MAE:", comparison.result["MAE", "ridge"], sep = " "), x = -3, y = 2.5)


# gbm plot
predictions.gbm = predict(gbmtuned, newdata = test)
df.gbm          = data.frame(cbind(test$logSalePrice, predictions.gbm))

gbm.plot = ggplot(df.gbm, aes(test$logSalePrice, predictions.gbm)) + geom_point() + geom_segment(x = -4, 
    y = -4, xend = 4, yend = 4, color = "red", size = 1.3) + stat_smooth(method = "lm", se = FALSE) + labs(title = "Plot of real logSalePrice against predicted values", 
    x = "logSalePrice", y = "gbmtuned predictions") + theme(axis.title = element_text(size = 16), plot.title = element_text(size = 16, 
    face = "bold")) + annotate("text", label = paste("MSE:", comparison.result["MSE", "gbm"], sep = " "), x = -3, y = 3) + 
    annotate("text", label = paste("MAE:", comparison.result["MAE", "gbm"], sep = " "), x = -3, y = 2.5)


# rf plot
predictions.rf = predict(rftuned, newdata = test)
df.rf          = data.frame(cbind(test$logSalePrice, predictions.rf))

rf.plot = ggplot(df.rf, aes(test$logSalePrice, predictions.rf)) + geom_point() + geom_segment(x = -4, 
    y = -4, xend = 4, yend = 4, color = "red", size = 1.3) + stat_smooth(method = "lm", se = FALSE) + labs(title = "Plot of real logSalePrice against predicted values", 
    x = "logSalePrice", y = "rftuned predictions") + theme(axis.title = element_text(size = 16), plot.title = element_text(size = 16, 
    face = "bold")) + annotate("text", label = paste("MSE:", comparison.result["MSE", "rf"], sep = " "), x = -3, y = 3) + annotate("text", 
    label = paste("MAE:", comparison.result["MAE", "rf"], sep = " "), x = -3, y = 2.5)


# plotting resulting graphs together
source("multiplot.R")

plot.list = list(lm.plot, fwd.plot, lasso.plot, ridge.plot, gbm.plot, rf.plot) # list of all graphs
plot.loop = seq(length(plot.list))
plot.loop = plot.loop[seq(length(plot.list)) %% 2 != 0]

pdf("Model_comparison.pdf", onefile = TRUE)           
for (i in plot.loop) {                                                         # looping through all models, creating a pdf with two plots per page for readability
    j = i + 1
    multiplot(plot.list[[i]], plot.list[[j]])  
}
dev.off()

