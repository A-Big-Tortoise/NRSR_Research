library("reticulate")
np <- import("numpy")

library("FMM")
set.seed(1115)

library("RColorBrewer")
library("ggplot2")
library("gridExtra")

scg0_path <- ("D:\\R\\R-Code\\FMM\\Dataset\\sim_5000_0_90_140_train.npy")
scg1_path <- ("D:\\R\\R-Code\\FMM\\Dataset\\sim_5000_0.1_90_140_train.npy")
scg8_path <- ("D:\\R\\R-Code\\FMM\\Dataset\\sim_5000_0.8_90_140_train.npy")

scg0_np <- np$load(scg0_path)
scg1_np <- np$load(scg1_path)
scg8_np <- np$load(scg8_path)

scg0_matrix <- as.matrix(scg0_np)
scg1_matrix <- as.matrix(scg1_np)
scg8_matrix <- as.matrix(scg8_np)


one_signal0 <-  scg0_matrix[1, 1: 992]
time <- seq(from = 0, to = 9.91, by = 0.01)

one_signal0 <-  scg0_matrix[1, 1: 60]
time <- seq(from = 0, to = .59, by = 0.01)


plot(time, one_signal0, cex = 0.5, xlab = "Time", ylab = "Magnitude", main = "SCG Signal with no Noise")

# result_scg0 <- fitFMM(vData=one_signal0, nPeriods = 16, timePoints=time, nback=4, betaOmegaRestrictions=c(1,1,2,2))
# result_scg0 <- fitFMM(vData=one_signal0, nPeriods = 16, nback=4, betaOmegaRestrictions=c(1,1,2,2))
# result_scg0 <- fitFMM(vData=one_signal0, nPeriods = 16, nback=4)
result_scg0 <- fitFMM(vData=one_signal0, nback=3)


titleText <- "SCG Signals"

result_scg = result_scg0

defaultrFMM2 <- plotFMM(result_scg, use_ggplot2 = TRUE, textExtra = titleText) + 
  theme(plot.margin=unit(c(1,0.25,1.3,1), "cm"))

comprFMM2 <- plotFMM(result_scg, components=TRUE, use_ggplot2 = TRUE, textExtra = titleText) + 
  theme(plot.margin=unit(c(1,0.25,0,1), "cm"))

grid.arrange(defaultrFMM2, comprFMM2, nrow = 1)


