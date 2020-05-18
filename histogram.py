import pandas as pd
import matplotlib.pyplot as plt
col_list = ["center",	"left",	"right",	"steering",	"throttle",	"brake", 	"speed"]
df = pd.read_csv("./driving_log.csv", usecols=col_list)
print(df["steering"])
fig = plt.hist(df["steering"],normed=0)
plt.title('Steering value histograms')
plt.xlabel("Steering value")
plt.ylabel("Frequency")
plt.savefig("Histograms.png")
