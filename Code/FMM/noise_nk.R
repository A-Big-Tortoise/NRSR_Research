library("reticulate")
np <- import("numpy")

library("FMM")
set.seed(1115)

library("RColorBrewer")
library("ggplot2")
library("gridExtra")

scg0_path <- ("D:\\R\\R-Code\\FMM\\Dataset\\sim_5000_0_90_140_train.npy")
scg1_path <- ("D:\\R\\R-Code\\FMM\\Dataset\\sim_5000_0.1_90_140_train.npy")

scg0_np <- np$load(scg0_path)
scg1_np <- np$load(scg1_path)
scg0_matrix <- as.matrix(scg0_np)
scg1_matrix <- as.matrix(scg1_np)

one_signal0 <-  scg0_matrix[1, 1: 60]
one_signal1 <-  scg1_matrix[1, 1: 60]

time <- seq(from = 0, to = .59, by = 0.01)


plot(time, one_signal1, cex = 0.5, xlab = "Time", ylab = "Magnitude", main = "SCG Signal with no Noise")

result_scg0_1 <- fitFMM(vData=one_signal0, nPeriods = 1, nback=4)
result_scg0_2 <- fitFMM(vData=one_signal1, nPeriods = 1, nback=4)


titleText <- "SCG Signals"

result_plt1 = result_scg0_1
result_plt2 = result_scg0_2

defaultrFMM1 <- plotFMM(result_plt1, use_ggplot2 = TRUE, textExtra = titleText) + 
  theme(plot.margin=unit(c(1,0.25,1.3,1), "cm"))

comprFMM1 <- plotFMM(result_plt1, components=TRUE, use_ggplot2 = TRUE, textExtra = titleText) + 
  theme(plot.margin=unit(c(1,0.25,0,1), "cm"))

defaultrFMM2 <- plotFMM(result_plt2, use_ggplot2 = TRUE, textExtra = titleText) + 
  theme(plot.margin=unit(c(1,0.25,1.3,1), "cm"))

comprFMM2 <- plotFMM(result_plt2, components=TRUE, use_ggplot2 = TRUE, textExtra = titleText) + 
  theme(plot.margin=unit(c(1,0.25,0,1), "cm"))

grid.arrange(defaultrFMM1, comprFMM1, defaultrFMM2, comprFMM2, nrow = 2)


