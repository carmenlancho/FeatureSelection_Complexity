from scipy.io.arff import loadarff
import numpy as np
import pandas as pd

def arff_to_pd(path):
  data = loadarff(path)
  raw_data, meta_data = data

  count = 0
  for i in range(raw_data.shape[0]):
    count += len(raw_data[i][1])

  cols = []
  for i in range(1,len(raw_data[0][1][0])+1):
    cols.append("f{0}".format(i))

  mol_names = []
  labels = []
  data2d = np.zeros([count, len(cols)])
  m = 0

  for i in range(raw_data.shape[0]):
    for j in range(len(raw_data[i][1])):
      mol_names.append(raw_data[i][0].decode())
      labels.append(int(raw_data[i][2].decode()))

      row_data = []

      for col_name in cols:
          row_data.append(raw_data[i][1][j][col_name])

      data2d[m] = row_data
      m += 1

  df = pd.DataFrame(data2d, columns = cols)
  df2 = df.assign(molecul_name = mol_names)
  df2_cols = df2.columns.tolist()
  df2_cols = df2_cols[-1:] + df2_cols[:-1]
  df2 = df2[df2_cols]
  df2 = df2.assign(label = labels)


  return df2


from ucimlrepo import fetch_ucirepo

# fetch dataset
musk_version_2 = fetch_ucirepo(id=75)

# data (as pandas dataframes)
X = musk_version_2.data.features
y = musk_version_2.data.targets

# metadata
print(musk_version_2.metadata)

# variable information
print(musk_version_2.variables)

df_musk2 = pd.DataFrame(X)
df_musk2['y'] = y
df_musk2.to_csv('musk2.csv',index=False)

from ucimlrepo import fetch_ucirepo

# fetch dataset
chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22)

# data (as pandas dataframes)
X = chess_king_rook_vs_king_pawn.data.features
y = chess_king_rook_vs_king_pawn.data.targets

# metadata
print(chess_king_rook_vs_king_pawn.metadata)

# variable information
print(chess_king_rook_vs_king_pawn.variables)

df_chess = pd.DataFrame(X)
df_chess['y'] = y
df_chess.to_csv('chess_kr_kp.csv',index=False)
