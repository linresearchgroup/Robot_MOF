# BO for MOF_crystallinity project
install.packages("smoof")
install.packages("checkmate")
install.packages("ParamHelpers")
install.packages("mlr")
install.packages("mmpf")
install.packages("mlrMBO")
library(mlr)
library(mlrMBO)


suppressWarnings({suppressMessages({  
  library(mlrMBO)  
	library(ggplot2)  
})})  

# Define directory of file as path  
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  
# Load  file
df = read.csv("MOF_crystal.csv")

# Define search space
ss = makeParamSet(  
  makeDiscreteParam("Ratio", values = c("0.125", "0.08333", "0.0625", "0.05")),
  makeIntegerParam("Volume", lower = 0.02, upper = 0.08, trafo = function(x) x+0.01),
  makeIntegerParam("Voltage", lower = 8, upper = 12, trafo = function(x) x+0.2),
  makeIntegerParam("Time", lower = 5, upper = 15, trafo = function(x) x+1)
)

# Define surrogate model configurations
ctrl = makeMBOControl(y.name = "Crystallinity")

# Define acquisition function and focus search configuration
ctrl = setMBOControlInfill(ctrl, opt = "focussearch", opt.focussearch.maxit = 10,
                           opt.focussearch.points = 10000, crit = makeMBOInfillCritEI())

# Construct surrogate model
suppressMessages({opt.state = initSMBO(par.set = ss,  
                                       design = subset(df, select = -c(estimated,estimatedUpper)),   
                                       control = ctrl, minimize = FALSE, noisy = TRUE)})  


# Suggest next cycle
cat("Suggested parameters:\n")
suggest = suppressWarnings({proposePoints(opt.state)})