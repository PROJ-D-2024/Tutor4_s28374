import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/customers-10000.csv')


# Пример кодирования столбца "Country"


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Преобразуем обратно в DataFrame
data = pd.DataFrame(data_scaled, columns=data.columns)





