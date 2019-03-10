import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from datetime import datetime
import itertools
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse

plt.style.use("seaborn-dark")

PATH = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"
def paser(x):
    return datetime.strptime(x, "%Y")

df = pd.read_csv(PATH, index_col=0,
                parse_dates=[0],
                date_parser=paser,
                verbose=True)


tsa.graphics.plot_pacf(df, lags=36)

def model_selection(
    series,
    param_set={
        "P": range(1, 9),
        "D": range(1, 3),
        "Q": range(1, 9)
    }
):
    order_lst = itertools.product(
        *[param_set[i] for i in ["P", "D", "Q"]]
    )
    records = list()  # List to store training results.
    fails = 0
    for order in tqdm(order_lst):
        try:
            model = tsa.ARIMA(series, order=order, freq="AS-JAN")
            model_fit = model.fit(disp=0)
            records.append({"order": order, "result": model_fit})
        except:
            fails += 1
            continue
    
    records.sort(key=lambda x: x["result"].aic)
    print(f"Total failed orders: {fails}")
    return records

records = model_selection(df)

for r in records:
    print(f"Order={r['order']}, AIC={r['result'].aic}")

def arima_test(
    order: tuple,
    train: np.ndarray,
    test: np.ndarray
):
    hist = list(train)
    pred = list()
    for t in tqdm(range(len(test))):
        model = tsa.ARIMA(hist, order=order)
        model_fit = model.fit(disp=0)
        f1, _, _ = model_fit.forecast(steps=1)
        pred.append(f1[0])
        hist.append(f1[0])  # Rolling prediction
    print(f"MSE={mse(pred, test)}")
    pred = np.array(pred)
    pred = pred.reshape(-1, 1)
    return pred


N = 240
train, test = df[:N].values, df[N:].values
pred = arima_test((5, 1, 4), train, test)
residual = test - pred
plt.plot(pred, label="Prediction")
plt.plot(test, label="Actual")
plt.legend()
plt.show()

plt.plot(residual, label="Residual")
plt.legend()
plt.show()
