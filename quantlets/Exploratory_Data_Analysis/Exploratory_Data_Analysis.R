# Install and load packages
libraries = c("ggplot2", "xtable")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
    install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# Read in dataing Data:
data = read.csv("train.csv")


# Analysis of Target Variable

# Histogram of Target Variable
price.hist = ggplot(data, aes(x = SalePrice)) + geom_histogram(aes(y = ..density..), bins = 20, 
    colour = "black", fill = "white") + geom_density(alpha = 0.2, fill = "#FF6666") + ggtitle("Distribution of Sale Price vs. Normal Distribution") + 
    xlab("Sale Price") + stat_function(fun = dnorm, args = list(mean = mean(data$SalePrice, 
    na.rm = TRUE), sd = sd(data$SalePrice, na.rm = TRUE)), col = "blue", size = 2)
ggsave(filename = "PriceHist.pdf")
# Histogram and QQ Plot of Log Prices
logprice.hist = ggplot(data, aes(x = log(SalePrice))) + geom_histogram(aes(y = ..density..), 
    bins = 20, colour = "black", fill = "white") + geom_density(alpha = 0.2, fill = "#FF6666") + 
    ggtitle("Distribution of  Log Sale Price vs. Normal Distribution") + xlab("Log Sale Price") + 
    stat_function(fun = dnorm, args = list(mean = mean(log(data$SalePrice), na.rm = TRUE), sd = sd(log(data$SalePrice), 
        na.rm = TRUE)), col = "blue", size = 2)
ggsave(filename = "LogPriceHist.pdf")
# Standardised Log Prices
stdprice.hist = ggplot(data, aes(x = (log(SalePrice) - mean(log(SalePrice), na.rm = TRUE))/sd(log(SalePrice), 
    na.rm = TRUE))) + geom_histogram(aes(y = ..density..), bins = 20, colour = "black", fill = "white") + 
    geom_density(alpha = 0.2, fill = "#FF6666") + ggtitle("Distribution of Standardised Log Sale Price vs. Normal Distribution") + 
    xlab("Standardised Log Sale Price") + stat_function(fun = dnorm, col = "blue", size = 2)
ggsave(filename = "StdPriceHist.pdf")

stdprice.qq = ggplot(data, aes(sample = (log(SalePrice) - mean(log(SalePrice), na.rm = TRUE))/sd(log(SalePrice), 
    na.rm = TRUE))) + stat_qq() + geom_abline(slope = 1, intercept = 0, col = "red") + ggtitle("QQ-Plot of Standardised Log Sale Price") + 
    xlab("Standardised Log Sale Price")
ggsave(filename = "StdPriceQQ.pdf")

# Get Column Classes:
colclasses = sapply(data, class)
table(colclasses)

# Seperate data in numeric and categoric variables for further analysis
numeric.data = data[, names(colclasses[colclasses != "factor"])]
categoric.data = data[, names(colclasses[colclasses == "factor"])]

# Drop Id Variable since it is unnecessary here
numeric.data$Id = NULL
# Exclude Target Variable
numeric.data$SalePrice = NULL

# Define Functions for variable overview
getmode = function(x) {
    x = x[!is.na(x)]
    unique(x)[which.max(tabulate(match(x, unique(x))))]
}
getmodefreq = function(x) {
    mean(x == getmode(x), na.rm = TRUE)
}
getlevelcount = function(x) {
    x = x[!is.na(x)]
    length(unique(x))
}
#These are just wrappers around existing functions setting na.rm = TRUE
meanwrapper       = function(x) mean(x, na.rm = TRUE)
medianwrapper     = function(x) median(x, na.rm = TRUE)
sdwrapper         = function(x) sd(x, na.rm = TRUE)

# Create Overview table for categoric variables:
categoric.overview = data.frame(NACount = colSums(sapply(categoric.data, is.na)), LevelCount = sapply(categoric.data, 
    FUN = getlevelcount), Mode = sapply(categoric.data, FUN = getmode), ModeFrequency = sapply(categoric.data, 
    FUN = getmodefreq))
categoric.overview_latex = xtable(categoric.overview)
print(categoric.overview_latex, file = "categoric.overview.tex")

# Create Overview table for numericvariables Create Overview table for categoric variables:
numeric.overview = data.frame(NACount = colSums(sapply(numeric.data, is.na)), LevelCount = sapply(numeric.data, 
    FUN = getlevelcount), Mode = sapply(numeric.data, FUN = getmode), ModeFrequency = sapply(numeric.data, 
    FUN = getmodefreq), Mean = sapply(numeric.data, FUN = meanwrapper), Median = sapply(numeric.data, 
    FUN = medianwrapper), SD = sapply(numeric.data, FUN = sdwrapper))
numeric.overview_latex = xtable(numeric.overview)
print(numeric.overview_latex, file = "numeric.overview.tex")
