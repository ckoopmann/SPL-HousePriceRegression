# rm(list = ls())
#setwd('quantlets/Random_Forest')
# Cross Validation Parameter
ptrain      = 0.85
nfolds      = 5
nrepeats    = 1
# Tuning Parameter Values for Random Forest
mtry.tuning = c(10, 20, 30, 40, 50)
# Tuning Paramter Values for Gradient Boosting Machine
ntrees.tuning     = c(100, 500, 1000)
intdepth.tuning   = 1
shrinkage.tuning  = seq(0.01, 0.15, by = 0.01)
minobs.tuning     = 10
# install and load packages
libraries = c("ggplot2", "randomForest", "caret", "doParallel", "gbm", "xtable")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
    install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)
#Read in training data
df = read.csv("train_preprocessed.csv")
# set rownumbers in dataframe to NULL
df$X  = NULL
df$Id = NULL
# Simple Random Forest Model without parameter tuning for variable importance
rf                                  = randomForest(logSalePrice ~ ., data = df)
importance                          = rf$importance[order(rf$importance, decreasing = TRUE), , drop = FALSE]
importance.normalized               = as.data.frame(importance/importance[1])
names(importance.normalized)[1]     = "Normalized_Importance"
importance.normalized$variable      = row.names(importance.normalized)
# Plot Variable Importance
importance.plot = ggplot(data = importance.normalized[1:10, ], aes(x = variable, y = Normalized_Importance)) + geom_col() + 
    scale_x_discrete(limits = importance.normalized[1:10, ]$variable) + ggtitle("Random Forest Variable Importance Top 10") + 
    xlab("Variable") + ylab("Normalized Variable Importance")
ggsave(importance.plot, filename = "rf_imp.pdf", width = 20, height = 10, units = 'cm')
# Export Variable Importance as Latex Table
importance.normalized_latex = xtable(importance.normalized)
print(importance.normalized_latex, file = "importance.tex")
# Parameter Tuning based on caret package Choose Valdation Method
CrossValidation = trainControl(method = "repeatedcv", number = nfolds, p = ptrain, repeats = nrepeats)
# Create Random Forest Tuning Grid:
rfGrid = expand.grid(mtry = mtry.tuning)
#Determine Number of Cores to use (All Cores but one)
cores       = max(c(detectCores() - 1, 1))
#Set Up Cluster for parallel processing
cl          = makeCluster(cores)
registerDoParallel(cl)
set.seed(123)
#Tune Model
rftuned = train(logSalePrice ~ ., data = df, method = "rf", importance = TRUE, trControl = CrossValidation, verbose = TRUE, 
    tuneGrid = rfGrid)
stopCluster(cl)
# Save RF-Model
save(rftuned, file = "rf.RData")
# Analyse Random Forest Results
rfresults         = rftuned$results
# Export Results as Latex Table
rfresults_latex   = xtable(rfresults)
print(rfresults_latex, file = "rfresults.tex")
# Plot RMSE vs mtry
rfrmse = ggplot(data = rfresults, aes(x = mtry, y = RMSE)) + geom_line() + ggtitle("Root Mean Squared Error vs. Tuning Parameter mtry") + 
    xlab("mtry") + ylab("RMSE")
ggsave(rfrmse, filename = "rf_rmse.pdf", width = 20, height = 10, units = 'cm')
# Plot RSquared vs mtry
rfrsq = ggplot(data = rfresults, aes(x = mtry, y = Rsquared)) + geom_line() + ggtitle("R Squared vs. Tuning Parameter mtry") + 
    xlab("mtry") + ylab("R Squared")
ggsave(rfrsq, filename = "rf_rsq.pdf", width = 20, height = 10, units = 'cm')
# Tune Gradient Boosting Machine Create Tuning Grid
gbmGrid     = expand.grid(n.trees = ntrees.tuning, interaction.depth = intdepth.tuning, shrinkage = shrinkage.tuning, n.minobsinnode = minobs.tuning)
# Start Cluster to paralellize
cl          = makeCluster(cores)
registerDoParallel(cl)
set.seed(123)
#Tune Model
gbmtuned = train(logSalePrice ~ ., data = df, method = "gbm", trControl = CrossValidation, verbose = TRUE, tuneGrid = gbmGrid)
stopCluster(cl)
plot(gbmtuned)
# Save GBM-Model
save(gbmtuned, file = "gbm.RData")
gbmresults              = gbmtuned$results
gbmresults$n.trees      = as.factor(gbmresults$n.trees)
# Export GBM Tuning Results as latex table Export Results as Latex Table
gbmresults_latex = xtable(gbmresults)
print(gbmresults_latex, file = "gbmresults.tex")
# Plot RMSE vs. shrinkage for each value of n.trees
gbmrmse = ggplot(data = gbmresults, aes(x = shrinkage, y = RMSE, colour = n.trees)) + geom_line() + ggtitle("Root Mean Squared Error vs. Tuning Parameter Shrinkage") + 
    xlab("Shrinkage") + ylab("RMSE")
ggsave(gbmrmse, filename = "gbm_rmse.pdf", width = 20, height = 10, units = 'cm')
# Plot RSquared vs. shrinkage for each value of n.trees
gbmrsq = ggplot(data = gbmresults, aes(x = shrinkage, y = Rsquared, col = n.trees)) + geom_line() + ggtitle("R Squared vs. Tuning Parameter Shrinkage") + 
    xlab("Shrinkage") + ylab("R Squared")
ggsave(gbmrsq, filename = "gbm_rsq.pdf", width = 20, height = 10, units = 'cm')