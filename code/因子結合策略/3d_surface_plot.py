import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# test = pd.read_csv('C:/Users/user/iplab/stock_factor_lab/code/因子結合策略/buy_at_low_df.csv', encoding='utf-8-sig')
# test = pd.read_csv('C:/Users/user/iplab/stock_factor_lab/Data/n_bband_df.csv', encoding='utf-8-sig')


test = test[['MA', 'std', 'CAGR']]

df_unique = test.groupby(['MA', 'std']).mean().reset_index()
df_unique.columns = ['MA','std','CAGR']

# 創建網格點
MA = df_unique['MA'].unique()
std = df_unique['std'].unique()
MA, std = np.meshgrid(MA, std)

# 創建 CAGR 的網格點
CAGR = df_unique.pivot(index='std', columns='MA', values='CAGR').values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 繪製曲面
surf = ax.plot_surface(MA, std, CAGR, cmap='summer')

# 添加軸標籤和標題
ax.set_xlabel('MA')
ax.set_ylabel('std')
ax.set_zlabel('CAGR[%]',  rotation=90)  # 將 Z 軸標籤設置為 CAGR 的百分比形式
ax.set_title('3D Surface Plot of MA, std, and CAGR')

# 將 Z 軸標籤格式化為百分比形式
formatter = FuncFormatter(lambda x, _: '{:.1%}'.format(x))
ax.zaxis.set_major_formatter(formatter)

# 添加顏色條
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()