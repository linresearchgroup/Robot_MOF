# Partial Dependence
# Load File  
df = read.csv("MOF_crystallinity experiments.csv") 

# Construct regression model
surr.rf = makeLearner(cl = "regr.randomForest")

# Create regression task
task = makeRegrTask(data=df, target='crystallinity')

# Train regression model
fit.regr = train(surr.rf, task)

# Output PD results
pd = generatePartialDependenceData(fit.regr, task, {"Ratio"})

# Save PD results
write.csv(pd[['data']], 'PD_Ratio.csv')
