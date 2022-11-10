import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x="total_bill", y="sex", orientation='h')

fig.write_image('img.png')