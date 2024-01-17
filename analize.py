import matplotlib.pyplot as plt 
import seaborn as sn 
from validation import gps_merge

plt.figure(figsize=(6,4))
plt.title("Countplot Content Rating")
sn.countplot(gps_merge, x=gps_merge['Content Rating'])
plt.show()


plt.figure(figsize=(6,4))
plt.title("hist rating")
sn.histplot(gps_merge['Rating'])
plt.show()

plt.figure(figsize=(6,4))
plt.title("Countplot of Sentiment")
sn.countplot(gps_merge, x=gps_merge['Sentiment'])
plt.show()