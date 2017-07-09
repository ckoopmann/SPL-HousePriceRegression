#regression postestimation
#function to plot residuals versus fitted values
rvfplot.gg = function(model) {
      ggplot(model, aes(.fitted, .resid)) + geom_point() + stat_smooth(method = "loess") + 
            geom_hline(yintercept = 0, col = "red", linetype = "dashed") + xlab("Fitted values") + 
            ylab("Residuals") + ggtitle("Residual vs Fitted Plot") + theme_bw()
}

pdf(file = "rvfplot.pdf", width = 8, height = 8)
rvfplot.gg(lm.fit)
dev.off()

#function to make a qqplot
qqplot.gg = function(model) {
      vec   = scale(model$residuals)
      y     = quantile(vec[!is.na(vec)], c(0.25, 0.75))
      x     = qnorm(c(0.25, 0.75))
      slope = diff(y)/diff(x)
      int   = y[1L] - slope * x[1L]
      d     = data.frame(resids = vec)
      ggplot(d, aes(sample = resids)) + stat_qq() + geom_abline(slope = slope, intercept = int) + 
            xlab("Theoretical Quantiles") + ylab("Standardized Residuals") + ggtitle("Normal Q-Q") + 
            theme_bw()
}

pdf(file = "qqplot.pdf", width = 8, height = 8)
qqplot.gg(lm.fit)
dev.off()