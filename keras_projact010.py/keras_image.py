import pandas as pd

pathlist = ["D:/test/choiminsik"]
data = pd.DataFrame(data= pathlist)
data.to_csv("D:/csv")