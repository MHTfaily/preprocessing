from PreProcessing import PreProcessingTweets
import pandas as pd

data = pd.read_csv("train_data.csv").iloc[:200,]
import time
t1=time.time()
pre = PreProcessingTweets(data)
pre.clean()
print("For 200 tweets it took "+str(round((time.time()-t1)/60, 2))+"min")