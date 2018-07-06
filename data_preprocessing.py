import numpy as np
import pandas as pd
from sklearn import preprocessing


class_types = ['hs92','hs96','hs02','hs07']

for s in class_types:
    # Criando dicionário que mapeia codigo hs as 21 maiores classes
    df = pd.read_csv('dados/products_{:s}.tsv'.format(s), sep='\t',dtype={'id': np.unicode_, s:np.unicode_})
    df = df[df[s].str.len() == 4]
    df.iloc[:,0] = df.iloc[:,0].apply(lambda x:x[:2])
    df = df.set_index(s)
    class_map = df.to_dict()['id']

    # Leitura dos dados de importação e exportação
    data = pd.read_csv('dados/{:s}_4.tsv'.format(s), sep='\t', dtype={s:np.unicode_}, usecols=[0,1,2,3,4]).fillna(0)

    # Agregando os dados nas 21 maiores classses
    data[s] = data[s].map(class_map)
    data = data.groupby(['year','origin',s], as_index=False).sum()

    # Calculo do RCA

    # Importação e exportação da classe em cada ano
    hs_sum = data.groupby(['year', s]).sum()
    data = data.join(hs_sum, on=['year', s], rsuffix='_hs')

    # Importação e exportação do país em cada ano
    country_sum = data.groupby(['year', 'origin']).sum()
    data = data.join(country_sum, on=['year', 'origin'], rsuffix='_c')

    # Importaão e exportação total no ano
    total_sum = data.groupby(['year']).sum()
    data = data.join(total_sum, on=['year'], rsuffix='_total')

    # RCA de exportação
    data['export_rca'] = (data['export_val']/data['export_val_c'])/(data['export_val_hs']/data['export_val_total'])
    data.drop(columns=['export_val','export_val_hs','export_val_total', 'export_val_c'], inplace=True)

    # RCA de importação
    data['import_rca'] = (data['import_val']/data['import_val_hs'])/(data['import_val_c']/data['import_val_total'])
    data.drop(columns=['import_val','import_val_hs','import_val_total', 'import_val_c'], inplace=True)

    # Formatando para tabela de features
    data = data.pivot_table(index=['year','origin'], columns=s, values=['import_rca','export_rca']).fillna(0)

    # Removendo origens inválidas
    data.drop(index=['xxa','xxb', 'xxc', 'xxd','xxe','xxf','xxg','xxh','xxi'], level='origin', errors='ignore', inplace=True)

    # Normalizando os valores para ficar entre 0 e 1
    scaler = preprocessing.MinMaxScaler()
    for year in data.index.get_level_values(level='year').unique():
        data.loc[year][data.columns] = scaler.fit_transform(data.loc[year])

    # Salvando dados já préprocessados
    data.to_pickle('data_{:s}.pkl'.format(s))
