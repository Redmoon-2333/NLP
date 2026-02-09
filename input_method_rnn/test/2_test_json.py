import pandas as pd

dfjo=pd.DataFrame(
    dict(A=range(1,4),B=range(4,7),C=range(7,10)),
    columns=["A","B","C"],
    index=list("xyz")
)

print(dfjo)

dfjo.to_json("df1.json",orient="columns")
dfjo.to_json("df2.json",orient="index")
dfjo.to_json("df3.json",orient="values")
dfjo.to_json("df4.json",orient="split")
dfjo.to_json("df5.json",orient="records")
dfjo.to_json("df6.json",orient="records",lines=True)
