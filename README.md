# Análise do Perfil de Exportações dos Paı́ses

Trabalho relativo a matéria de Aprendizado de Máquina lecionada na UMFMG. Dados obtidos no site 
https://atlas.media.mit.edu/en/resources/data/
## Resumo
Nesse trabalho algumas técnicas de aprendizado não supervisionado foram aplicadas em um conjunto de dados de exportações de paises.
O intuito era encontrar algum agrupamento natural que explicasse a dinâmica de comercio internacional através de um número 
reduzido de estratégias de exportações. Os dados de exportações são relativos a classes de produtos de acordo com 
a classificação Harmonic "Harmonized Commodity Description and Coding Systems". As principais técnicas abordados nesse estudo
são:
* Principal component analysis (PCA)
* t-distributed stochastic neighbor embedding (t-SNE)
* Agrupamento Hierárquico
* K-means

## Estrutura
### dados
Arquivos baixados atrvés do site https://atlas.media.mit.edu/en/resources/data/. Para rodar o script de pre-processamento é
necessário ter essa pasta.
### data_preprocessing.py
Script que faz o preprocessamento dos arquivo e salva em formato de pandas.DataFrame. Para que o código presente no notebook
funcione é necessário rodar esse arquivo
### clustering.py
Arquivo importado dentro do notebook com algumas funções utilizadas para fazer o agrupamento dos países.
### Análise_países.ipynv
Notebook com passo a passo para geração das figuras presentes no relatório.
### Relatório.pdf
Relatório final do trabalho, possui todas as análises e conclues.
