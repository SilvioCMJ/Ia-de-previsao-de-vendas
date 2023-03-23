import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegressio
from sklearn.metrics import r2_score

#importando tabela

tabela = pd.read_csv('barcos_ref.csv')
correlacao = (tabela.corr()[['Preco']])
print(correlacao)

#criando grafico
sns.heatmap(correlacao, cmap="Greens",annot=True)
plt.show()

#criar ia

y = tabela['Preco']
x = tabela.drop('Preco', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

regre_linear = LinearRegression()
model_arvore = RandomForestRegressor()

regre_linear.fit(x_treino, y_treino)
model_arvore.fit(x_treino, y_treino)

#interpretação de resultados
#escolher o melhor modelo
previsao_linear = regre_linear.predict(x_teste)
previsao_arvore = model_arvore.predict(x_teste)

print(r2_score(y_teste,previsao_linear))
print(r2_score(y_teste,previsao_arvore))

#visulaizar previsoes
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar['ArvoreDecisao'] = previsao_arvore
tabela_auxiliar['RegressaoLinear'] = previsao_linear

sns.lineplot(data=tabela_auxiliar)
plt.show()

#fazendo novas previsoes usando ia
tabela_nova = pd.read_csv("novos_barcos.csv")
display(tabela_nova)

previsao =  model_arvore.predict(tabela_nova)
print(previsao)





