import pandas as pd
from style import highlight

def load_dataframe(path):
  dataframe_name = path.split('/')[-1]
  print('Carregando dataset {}...'.format(highlight(dataframe_name)))
  df = pd.read_csv(path)

  labels = df['label'].to_numpy()
  df = df.drop('label', axis=1)

  return df, labels
