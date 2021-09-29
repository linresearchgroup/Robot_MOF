# Partial Dependence
install.packages("mlr")
install.packages("mmpf")
library(mlr)

# Load File  
df = read.csv("MOF_crystal.csv") 

# Construct regression model
surr.rf = makeLearner(cl = "regr.randomForest")

# Create regression task
task = makeRegrTask(data=df, target='Crystallinity')

# Train regression model
fit.regr = train(surr.rf, task)

# Output PD results
pd = generatePartialDependenceData(fit.regr, task, {"Time"})

# Save PD results
write.csv(pd[['data']], 'PD_Time.csv')
