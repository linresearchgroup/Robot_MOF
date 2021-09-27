# Load library
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'ticks', color_codes = True)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load datasets
df = pd.read_excel('MOF_crystallinity experiments.xlsx').drop(['No'], axis =1)

# Histogram
plt.figure(figsize = (10, 8))
plt.xlabel('Crystallinity', fontsize = 18)
plt.ylabel('Percentage', fontsize = 18)
sns.distplot(df.iloc[:, 4], axlabel = 'Crystallinity', label = 'Percentage')
plt.savefig('Histogram.png', dpi = 300)
plt.show()
# Heatmap
plt.figure(figsize = (10, 8))
ax = sns.heatmap(df.iloc[:, 0:4].corr(), cmap="YlGnBu", annot = True, vmin = 0, vmax = 1)
ax.set_ylim([4, 0])
plt.savefig('Heatmap.png', dpi = 600)
plt.show()


for rs in [14]:
    # Split the datasets into training and test datasets
    X = df.iloc[:, 0:4]
    y = df.iloc[:, 4]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        shuffle = True, 
                                                        random_state = rs)
    # MinMaxScaler
    mms = MinMaxScaler()
    X_train_scaled = mms.fit_transform(X_train)
    X_test_scaled = mms.transform(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    # Build MLP model
    mlp = Sequential()
    mlp.add(Dense(units = 8, activation = 'relu', input_dim = X.shape[1]))
    mlp.add(Dense(units = 8, activation = 'relu'))
    mlp.add(Dense(units = 1))
    mlp.compile(optimizer = 'rmsprop', 
                loss='mean_squared_error', 
                metrics = ['mae'])
    
    crystallinity_hist = mlp.fit(X_train_scaled, y_train, epochs = 1000, batch_size = 4)
    # Saving SSA model
    mlp.save('Crystallinity RS{}.h5'.format(rs))
    # Predicting results

    y_pred_test_mlp = mlp.predict(X_test_scaled)
    y_pred_test_mlp = y_pred_test_mlp.reshape(-1)


    test_array = pd.DataFrame({'y_test': np.array(y_test),
                               'y_pred_test':y_pred_test_mlp,
                               'AE': np.abs(y_test - y_pred_test_mlp),
                               'APE': np.abs((y_test - y_pred_test_mlp)/y_test)*100})       
    test_array.to_excel('Crystallinity {} Test RS{}.xlsx'.format(rs))       
   
    test_eval = pd.Series({'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_test_mlp)), 2),
                            'R^2': round(r2_score(y_test, y_pred_test_mlp), 2),
                            'MAE': round(sum(test_array['AE'])/len(y_test), 2),
                            'MAPE':round(sum(test_array['APE'])/len(y_test), 2)})
    df2 = pd.DataFrame({'Test': test_eval})
    # Draw figure for visualiation and comparison
    plt.figure(figsize = (8, 8))
    plt.scatter(y_test, y_pred_test_mlp, c='Tab:blue', alpha = 0.6, edgecolors = 'none',
                label= "Test (RMSE:{} R^2:{} MAE:{} MAPE:{})".format(df2.iloc[0, 0], df2.iloc[1, 0], df2.iloc[2, 0], df2.iloc[3,0]))
    plt.plot([0, 1], [0, 1], c ='orange', ls = '--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('y_true', fontsize = 18)
    plt.ylabel('y_pred', fontsize = 18)
    plt.title('Crystallinity RS{}'.format(rs), fontsize = 22)
    plt.legend(loc="lower right", frameon = False)
    plt.savefig('Crystallinity RS{}.png'.format(rs), dpi = 300)
    plt.show()

    crystallinity_hist_dic = crystallinity_hist.history
    loss = crystallinity_hist_dic['loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize = (10, 8))
    plt.plot(epochs, loss, 'bo', alpha = 0.6,
             label = 'Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')    
    plt.title('Training Loss', fontsize = 22)
    plt.legend()
    plt.savefig('Crystallinity Loss RS{}'.format(rs), dpi = 300)
    plt.show()
    
    df3 = pd.read_excel('pred_blank.xlsx').drop(['No'], axis =1)
    
    df_trans = mms.transform(df3)
    crystal_pred = mlp.predict(df_trans)
    df3['C'] = crystal_pred
    df3.to_excel('Crystallinity_pred.xlsx')


    
    
    
    
