import numpy as np
import pandas as pd
from numba import njit, prange, stencil
from pprint import pprint as pp
import matplotlib.pyplot as plt

df = pd.read_csv(
    '~/Downloads/EURUSD-2019-06.zip',
    compression='zip',
    header=None,
    names=['pair', 'timestamp', 'bid', 'ask'],
    usecols=['timestamp', 'bid', 'ask'],
    parse_dates=[0],
    index_col=0
)


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
def assign_bar(b):
    state = np.empty((b.shape[0], 5))
    state[0][3] = np.mean(b)
    state[0][2] = 1.e5
    n = b.shape[0]

    for i in prange(n):
        state[i][0] = state[i - 1][0] + b[i]  # cumsum
        state[i][1] = state[i - 1][1]         # T
        state[i][2] = state[i - 1][2]         # E_T
        state[i][3] = state[i - 1][3]         # E_b
        state[i][4] = state[i - 1][4]         # bar_start

        if abs(state[i][0]) >= state[i][2] * abs(state[i][3]):
            # T = (i - bar_start)
            state[i][1] = float(i - state[i][4])
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
            state[i][4] = i

    return state[:, 4]


df = df.assign(
    bar=df.index[
        assign_bar(
            df.mid_b.values
        ).astype(np.int)
    ]
)

df = df.reset_index()
df = df.set_index(['bar', 'timestamp'])

pp(df[575000:].groupby(by=['bar']).mean())

df[575000:].boxplot(column=['mid'], by='bar')
plt.show()
