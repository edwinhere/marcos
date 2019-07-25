import numpy as np
import pandas as pd
from numba import njit, prange, stencil
# from pprint import pprint as pp
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

currency = 'EURUSD'
years = range(2019, 2020)
months = range(5, 7)
df = pd.DataFrame()

for year in years:
    for month in months:
        print('~/Downloads/%s-%s-%02d.zip' % (currency, year, month))

        frame = pd.read_csv(
            '~/Downloads/%s-%s-%02d.zip' % (currency, year, month),
            compression='zip',
            header=None,
            names=['pair', 'timestamp', 'bid', 'ask'],
            usecols=['timestamp', 'bid', 'ask'],
            parse_dates=[0],
            index_col=0
        )

        df = df.append(frame)


@stencil
def tick_rule(p, lastb):
    delta_p = p[0] - p[-1]
    if delta_p == 0:
        return lastb
    else:
        lastb = abs(delta_p) / delta_p
        return lastb


df['mid'] = (df.bid + df.ask) / 2.0
df = df.assign(mid_b=tick_rule(df.mid.values, 0))
df.mid_b[0] = 1.0


@njit(parallel=True)
def assign_bar(timestamp, b):
    state = np.empty((b.shape[0], 6))
    state[:, 5] = timestamp
    state[0][4] = timestamp[0]
    state[0][3] = np.mean(b[:100])
    state[0][2] = 1.e4
    n = b.shape[0]

    for i in prange(n):
        state[i][0] = state[i - 1][0] + (
            b[i] * (state[i - 1][5] - state[i - 2][5])  # cumsum
        )
        state[i][1] = state[i - 1][1]         # T
        state[i][2] = state[i - 1][2]         # E_T
        state[i][3] = state[i - 1][3]         # E_b
        state[i][4] = state[i - 1][4]         # bar_start
        state[i][5] = state[i - 1][5]         # timestamp

        if abs(state[i][0]) >= state[i][2] * abs(state[i][3]):
            # T = (i - bar_start)
            state[i][1] = float(state[i][5] - state[i][4])
            # E_T = E_T + 0.5 * (T - E_T)
            state[i][2] = state[i - 1][2] + 0.5 * (
                state[i][1] - state[i - 1][2]
            )
            # E_b = E_b + 0.5 * (cumsum - E_b)
            state[i][3] = state[i - 1][3] + 0.5 * (
                state[i][0] - state[i - 1][3]
            )
            # cumsum = 0
            state[i][0] = 0
            # bar_start = i
            state[i][4] = state[i][5]

    return state[:, 4]


df = df.assign(
    bar=assign_bar(
        df.index.astype('int64').to_numpy(),
        df.mid_b.values
    )
)

df['bar'] = pd.to_datetime(df['bar'])

df = df.reset_index()
df = df.set_index(['bar', 'timestamp'])

ohlc = df.mid.groupby('bar').agg([
    ('open', 'first'),
    ('high', 'max'),
    ('low', 'min'),
    ('close', 'last')
])

fig = go.Figure(
    data=go.Ohlc(
        x=ohlc.index,
        open=ohlc.open,
        high=ohlc.high,
        low=ohlc.low,
        close=ohlc.close
    )
)

fig.show()
