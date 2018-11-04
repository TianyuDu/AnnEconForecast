from constants import *
from core.tools import *

df = load_dataset(UNRATE_DIR["MAC"])
df_dd = differencing(df, periods=1, order=1)
df_d1 = differencing(df, periods=1, order=2)