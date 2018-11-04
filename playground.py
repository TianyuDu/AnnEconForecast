from constants import *
from core.tools import *

df = load_dataset(UNRATE_DIR["MAC"])
df_dd = differencing(df, periods=1, order=1)
df_d1 = differencing(df, periods=1, order=2)

lags = [i for i in range(1, 5)]
X, y = gen_supervised(df, predictors=lags)
X, y = clean_nan(X, y)
