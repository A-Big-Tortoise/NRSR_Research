library("FMM")
set.seed(1115)

rfmm.data <- generateFMM(M=3, A = c(4,3,1.5,1), alpha = c(3.8,1.2,4.5,2),
            beta = c(rep(3,2),rep(1,2)),
            omega = c(rep(0.1,2),rep(0.05,2)),
            plot = TRUE, outvalues = TRUE,
            sigmaNoise = 0.3)

fit.rfmm <- fitFMM(vData = rfmm.data$y, timePoints = rfmm.data$t, nback = 4,
                   betaOmegaRestrictions = c(1, 1, 2, 2))

# Plot the fitted FMM model
library("RColorBrewer")
library("ggplot2")
library("gridExtra")
titleText <- "Simulation of four restricted FMM waves"

defaultrFMM2 <- plotFMM(fit.rfmm, use_ggplot2 = TRUE, textExtra = titleText) +
  theme(plot.margin=unit(c(1,0.25,1.3,1), "cm")) + ylim(-5, 6)

comprFMM2 <- plotFMM(fit.rfmm, components=TRUE, use_ggplot2 = TRUE,
  textExtra = titleText) +
  theme(plot.margin=unit(c(1,0.25,0,1), "cm")) +
  ylim(-5, 6) 

grid.arrange(defaultrFMM2, comprFMM2, nrow = 1)
