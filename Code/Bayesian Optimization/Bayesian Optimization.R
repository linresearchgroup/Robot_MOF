# BO for MOF_crystallinity project

suppressWarnings({suppressMessages({  
  library(mlrMBO)  
	library(ggplot2)  
})})  


# Define search space
ss = makeParaSet(
                            makeDiscreteParam("Ratio", values = c("0.125", "0.08333", "0.0625", "0.05"),
                            makeIntegerParam("Volume", lower = 0.02, upper = 0.08, trafo = function(x) x+0.01),
                            makeIntegerParam("Voltage", lower = 8, upper = 12, trafo = function(x) x+0.2),
                            makeIntegerParam("Time", lower = 5, upper = 15, trafo = function(x) x+1)
                            )


# Define surrogate model configurations
ctrl = makeMBOControl(y.name = "crystallinity")

# Define acquistion function and focus search config
ctrl = setMBOControlInfill(ctrl, opt = "focussearch", opt.focussearch.maxit = 10,
                                         opt.focussearch.points = 10000, crit = makeMBOInfillCtriEI())

# Load  file
setwd(firname(rstudioapi::getActiveDocumentContext()$path))
df = read.csv("MOF_crystallinity.csv")

# Construct surrogate model
suppressMessages({opt.state = initSMBO(par.set = ss,
                                                               design = subset(df, select = -c(estimated, estimatedUpper)),
                                                               control = ctrl, minimize = FALSE, noisy = TRUE)})

# Suggest next cycle
cat("Suggested parameters:\n")
suggest = suppressWarnings({proposePoints(opt.state)})