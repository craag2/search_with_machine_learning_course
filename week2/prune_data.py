import pandas as pd

data_path = '/workspace/datasets/fasttext/labeled_products.txt'
df = pd.read_csv(data_path,  header=None, sep='\t', names=['row'])
df['label'] = df['row'].apply(lambda x : x.split(' ',1)[0])
df['product'] = df['row'].apply(lambda x : x.split(' ',1)[1])
print(len(df))

df_filtered = df.groupby('label').filter(lambda x: len(x) >= 500)

print(df_filtered['label'].value_counts())

print(len(df_filtered))

df_filtered.to_csv('./../pruned_labeled_products.txt', header=False, sep='\t', index=False, mode='a')




