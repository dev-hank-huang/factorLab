import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Report():
    def __init__(self, stock_data, position, data) -> None:
        self.stock_data = stock_data
        self.position = position
        self.calc_return_table()

        ##############################
        self.cashflow = None
        self.daily_return = None
        ##############################

        try:
            self.benchmark = data.get('taiex:close')
            self.benchmark.index = pd.to_datetime(self.benchmark.index)

            self.daily_benchmark = rebase(self.benchmark\
                .dropna().reindex(self.stock_data.index, method='ffill') \
                .ffill())
        except:
            pass

    def display(self):
        from IPython.display import display
        
        # 計算
        fig = self.create_performance_figure(self.position)

        stats = self.get_stats()

        imp_stats = pd.Series({
         'annualized_rate_of_return':str(round(self.calc_cagr()*100, 2))+'%',
         'sharpe': str(self.calc_sharpe(self.stock_data['portfolio_returns'], nperiods=252)),
         'max_drawdown':str(round(self.calc_dd(self.stock_data['portfolio_returns']).min()*100, 2))+'%',
         'win_ratio':str(round(stats['win_ratio']*100, 2))+'%',
        }).to_frame().T
        imp_stats.index = ['']

        yearly_return_fig = self.create_yearly_return_figure()
        monthly_return_fig = self.create_monthly_return_figure()
        tree_map = self.display_reutrn_treemap()
        # trade_histogram = self.display_each_trade_histogram()
        top5_cum_return = self.display_topN_cum_return_yearly(self.trades)

        # show出來
        display(imp_stats)
        display(fig)
        display(yearly_return_fig)
        display(monthly_return_fig)
        display(tree_map)
        # display(trade_histogram)
        display(top5_cum_return)
        if hasattr(self, 'current_trades'):
            display(self.current_trades)

    def create_monthly_return_figure(self):
        import plotly.express as px
        monthly_table = pd.DataFrame(self.return_table).T
        monthly_table = round(monthly_table*100,1).drop(columns='YTD')
        # monthly_table.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/monthly_return.csv')

        fig = px.imshow(monthly_table.values,
                        labels=dict(x="month", y='year', color="return(%)"),
                        x=monthly_table.columns.astype(str),
                        y=monthly_table.index.astype(str),
                        text_auto=True,
                        color_continuous_scale='RdBu_r',

                        )

        fig.update_traces(
            hovertemplate="<br>".join([
                "year: %{y}",
                "month: %{x}",
                "return: %{z}%",
            ])
        )

        fig.update_layout(
            height = 550,
            width= 800,
            margin=dict(l=20, r=270, t=40, b=40),
            title={
                'text': 'monthly return',
                'x': 0.025,
                'yanchor': 'top',
            },
            yaxis={
                'side': "right",
            },
            coloraxis_showscale=False,
            coloraxis={'cmid':0}
        )

        return fig
    
    def create_yearly_return_figure(self):
        import plotly.express as px
        yearly_return = [round(v['YTD']*1000)/10 for v in self.return_table.values()]

        fig = px.imshow([yearly_return],
                        labels=dict(color="return(%)"),
                        x=list([str(k) for k in self.return_table.keys()]),
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        )

        fig.update_traces(
            hovertemplate="<br>".join([
                "year: %{x}",
                "return: %{z}%",
            ])
        )

        fig.update_layout(
            height = 120,
            width= 800,
            margin=dict(l=20, r=270, t=40, b=40),
            yaxis={
                'visible': False,
            },
            title={
                'text': 'yearly return',
                'x': 0.025,
                'yanchor': 'top',
            },
            coloraxis_showscale=False,
            coloraxis={'cmid':0}
            )

        return fig

    def create_performance_figure(self, position):

        """
        plot the performance figure

        Parameters
        ----------
        position : pd.DataFrame
            The position matrix where index is the date and column is the stock id.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The figure that contains the performance figure.

        Examples
        --------
            >>> report = Report()
            >>> position = pd.DataFrame()
            >>> report.create_performance_figure(position)
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objs as go
        # plot performance

        def diff(s, period):
            return (s / s.shift(period) - 1)

        drawdowns = self.calc_dd(self.stock_data['portfolio_returns'])

        # test
        # drawdowns.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/drawdown.csv')

        if hasattr(self, 'benchmark'):
            benchmark_drawdown = self.calc_dd(self.daily_benchmark['close'])
        performance_detail = self.stock_data.copy()

        # test
        print(performance_detail['portfolio_returns'])

        nstocks = self.stock_data['company_count']

        fig = go.Figure(make_subplots(
            rows=4, cols=1, shared_xaxes=True, row_heights=[2, 1, 1, 1]))
        fig.add_scatter(x=performance_detail.index, y=round(performance_detail['cum_returns'] - 1, 2),
                        name='strategy', row=1, col=1, legendgroup='performance', fill='tozeroy')
        fig.add_scatter(x=drawdowns.index, y=drawdowns, name='strategy - drawdown',
                        row=2, col=1, legendgroup='drawdown', fill='tozeroy')
        fig.add_scatter(x=performance_detail.index, y=diff(performance_detail['portfolio_returns'], 20),
                        fill='tozeroy', name='strategy - month rolling return',
                        row=3, col=1, legendgroup='rolling performance', )
        
        # benchmark
        if hasattr(self, 'benchmark'):
            fig.add_scatter(x=performance_detail.index, y=self.daily_benchmark['close'] / 100 - 1,
                            name='benchmark', row=1, col=1, legendgroup='performance', line={'color': 'gray'})
            fig.add_scatter(x=drawdowns.index, y=benchmark_drawdown, name='benchmark - drawdown',
                            row=2, col=1, legendgroup='drawdown', line={'color': 'gray'})
            fig.add_scatter(x=performance_detail.index, y=diff(self.daily_benchmark['close'], 20),
                            fill='tozeroy', name='benchmark - month rolling return',
                            row=3, col=1, legendgroup='rolling performance', line={'color': 'rgba(0,0,0,0.2)'})

        fig.add_scatter(x=nstocks.index, y=nstocks, row=4,
                        col=1, name='nstocks', fill='tozeroy')

        fig.update_layout(legend={'bgcolor': 'rgba(0,0,0,0)'},
                          margin=dict(l=60, r=20, t=40, b=20),
                          height=600,
                          width=800,
                          xaxis4=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=1,
                                           label="1m",
                                           step="month",
                                           stepmode="backward"),
                                      dict(count=6,
                                           label="6m",
                                           step="month",
                                           stepmode="backward"),
                                      dict(count=1,
                                           label="YTD",
                                           step="year",
                                           stepmode="todate"),
                                      dict(count=1,
                                           label="1y",
                                           step="year",
                                           stepmode="backward"),
                                      dict(step="all")
                                  ]),
                                  x=0,
                                  y=1,
                              ),
                              rangeslider={'visible': True, 'thickness': 0.1},
                              type="date",
                          ),
                          yaxis={'tickformat': ',.0%', },
                          yaxis2={'tickformat': ',.0%', },
                          yaxis3={'tickformat': ',.0%', },
                          )
        return fig

    def calc_dd(self, daily_return):
        '''
        計算Drawdown的方式是找出截至當下的最大累計報酬(%)除以當下的累計報酬
        所以用累計報酬/累計報酬.cummax()

        return:
            dd : 每天的drawdown (可用來畫圖)
            mdd : 最大回落
            start : mdd開始日期
            end : mdd結束日期
            days : 持續時間
        '''
        # drawdown = self.stock_data['portfolio_returns'].copy()
        drawdown = daily_return.copy()

        # Fill NaN's with previous values
        drawdown = drawdown.ffill()

        # Ignore problems with NaN's in the beginning
        drawdown[np.isnan(drawdown)] = -np.Inf

        # Rolling maximum
        if isinstance(drawdown, pd.DataFrame):
            roll_max = pd.DataFrame()
            for col in drawdown:
                roll_max[col] = np.maximum.accumulate(drawdown[col])
        else:
            roll_max = np.maximum.accumulate(drawdown)

        drawdown = drawdown / roll_max - 1.0

        return drawdown

    def calc_cagr(self):
        '''
        計算CAGR用 (最終價值/初始價值) ^ (1/持有時間(年)) - 1
        那其實 最終價值/初始價值 跟「累計回報」會差不多，
        所以公式可以變為:累計報酬^(1/持有時間(年))-1

        return:
            cagr : 年均報酬率
        '''
        def year_frac(start, end):
            """
            Similar to excel's yearfrac function. Returns
            a year fraction between two dates (i.e. 1.53 years).
            Approximation using the average number of seconds
            in a year.
            Args:
                * start (datetime): start date
                * end (datetime): end date
            """
            if start > end:
                raise ValueError("start cannot be larger than end")

            # obviously not perfect but good enough
            return (end - start).total_seconds() / (31557600)
        
        daily_prices = self.stock_data['portfolio_returns'].resample("D").last().dropna()

        start = daily_prices.index[0]
        end = daily_prices.index[-1]
        # return (daily_prices.iloc[-1] / daily_prices.iloc[0]) ** (1 / year_frac(start, end)) - 1
        return (safe_division(daily_prices.iloc[-1], daily_prices.iloc[0])) ** safe_division(1, year_frac(start, end)) - 1

    def calc_return_table(self):
        '''
        產生月報酬率表

        產生一個 dict，key 是年份，value 是一個 dict，key 是月份，value 是當月報酬率

        2019:
            1: 0.05
            2: 0.03
            3: 0.02
            ...
            YTD: 1.13

        2020:
            1: 0.05
            2: 0.03
            3: 0.02
            ...
            YTD: 1.13
        '''
        self.return_table = {}
        obj = self.stock_data['portfolio_returns']
        daily_prices = obj.resample("D").last().dropna()
        # M = month end frequency
        monthly_prices = obj.resample("M").last()  # .dropna()
        # A == year end frequency
        yearly_prices = obj.resample("A").last()  # .dropna()

        # let's save some typing
        dp = daily_prices
        mp = monthly_prices
        yp = yearly_prices

        monthly_returns = mp / mp.shift(1) -1
        mr = monthly_returns

        for idx in mr.index:
            if idx.year not in self.return_table:
                self.return_table[idx.year] = {
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                    10: 0,
                    11: 0,
                    12: 0,
                }
            if not np.isnan(mr[idx]):
                self.return_table[idx.year][idx.month] = mr[idx]
        # add first month
        fidx = mr.index[0]
        try:
            self.return_table[fidx.year][fidx.month] = float(mp.iloc[0]) / dp.iloc[0] - 1
        except ZeroDivisionError:
            self.return_table[fidx.year][fidx.month] = 0
        # calculate the YTD values
        for idx in self.return_table:
            arr = np.array(list(self.return_table[idx].values()))
            self.return_table[idx]["YTD"] = np.prod(arr + 1) - 1

    def calc_sharpe(self, returns, rf=0.0, nperiods=None, annualize=True):
        """
        Calculates the `Sharpe ratio <https://www.investopedia.com/terms/s/sharperatio.asp>`_
        (see `Sharpe vs. Sortino <https://www.investopedia.com/ask/answers/010815/what-difference-between-sharpe-ratio-and-sortino-ratio.asp>`_).
        If rf is non-zero and a float, you must specify nperiods. In this case, rf is assumed
        to be expressed in yearly (annualized) terms.
        Args:
            * returns (Series, DataFrame): Input return series
            * rf (float, Series): `Risk-free rate <https://www.investopedia.com/terms/r/risk-freerate.asp>`_ expressed as a yearly (annualized) return or return series
            * nperiods (int): Frequency of returns (252 for daily, 12 for monthly,
                etc.)
        """
        # if type(rf) is float and rf != 0 and nperiods is None:
        if isinstance(rf, float) and rf != 0 and nperiods is None:
            raise Exception("Must provide nperiods if rf != 0")

        er = to_excess_returns(returns, rf, nperiods=nperiods)
        std = np.std(er, ddof=1)
        res = np.divide(er.mean(), max(std, 0.000001))

        if annualize:
            if nperiods is None:
                nperiods = 1
            sharpe_ratio = res * np.sqrt(nperiods)
        else:
            sharpe_ratio = res

        # Round the result to two decimal places
        return round(sharpe_ratio, 2)
    
    def calc_ytd(self, daily_prices, yearly_prices):
        if len(yearly_prices) == 1:
            return daily_prices.iloc[-1] / daily_prices.iloc[0] - 1
        else:
            return daily_prices.iloc[-1] / yearly_prices.iloc[-2] - 1

    def get_stats(self):
        """
        計算策略的績效指標

        Args:

        Returns:
            (dict): { 'CAGR': float, 'daily_sharpe': float, 'max_drawdown': float, 'avg_drawdown': float, 'win_ratio': float, 'ytd': float }
        """
       
        obj = self.stock_data['portfolio_returns']
        daily_prices = obj.resample("D").last().dropna()
        # M = month end frequency
        monthly_prices = obj.resample("M").last()  # .dropna()
        # A == year end frequency
        yearly_prices = obj.resample("A").last()  # .dropna()

        # let's save some typing
        dp = daily_prices
        mp = monthly_prices
        yp = yearly_prices

        # # 計算每日報酬率 20240721
        # daily_returns = dp.pct_change().dropna()

        trades = self.trades.dropna()

        stats = {}
        # stats["daily_mean"] = dr.mean() * 252
        stats["CAGR"] = self.calc_cagr()
        
        stats['daily_sharpe'] = self.calc_sharpe(dp, nperiods=252)
        # stats['daily_sharpe'] = self.calc_sharpe(daily_returns, nperiods=252)

        stats['max_drawdown'] = self.calc_dd(self.stock_data['portfolio_returns']).min()
        stats['avg_drawdown'] = self.calc_dd(self.stock_data['portfolio_returns']).mean()
        stats['win_ratio'] = sum(trades['return'] > 0) / len(trades) if len(trades) != 0 else 0
        stats['ytd'] = self.calc_ytd(dp,yp)

        return stats

    def display_reutrn_treemap(self):
        """
        Display a treemap of cumulative returns by stock.

        This method displays a treemap showing the cumulative returns for each stock in the portfolio.
        The size of each rectangle is proportional to the cumulative return of the stock.
        The color of each rectangle is based on the cumulative return of the stock, with red indicating a negative
        return and blue indicating a positive return.

        Returns:
            None
        """
        df = self.trades.copy()

        # test 
        # df['return'].to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/return.csv')

        df["cum_return"] =  (1 + df['return']).groupby(df['stock_id']).cumprod() - 1
        df = df.groupby('stock_id').last()
        df = df.reset_index()

        fig = px.treemap(df, path=['stock_id'], values='cum_return',
                 title='Treemap of Cumulative Returns by Stock',
                 color='cum_return', color_continuous_scale='RdBu_r',color_continuous_midpoint=0,
                 custom_data=['stock_id', 'cum_return'],
                 width=1600,
                 height=800)
        fig.update_traces(textposition='middle center',
                        textfont_size=20,
                        texttemplate="%{label}<br>cum_return = %{customdata[1]:.2f}",
                        )

        fig.show()

    def display_each_trade_histogram(self, index = "return"):
        # index 有可能是:
        # position/period/return/mae/gmfe/bmfe/mdd/pdays
        # 假設你的dataframe名稱是trades_history
        
        df = self.current_trades
        # 將return欄位的數值乘以100
        if(index == "position" or index == "return" or index == "mae" or index == "gmfe" or index == "bmfe" or index == "mdd" ):
            df[index] = df[index] * 100

        # 繪製直方圖
        plt.figure(figsize=(10, 6))
        data = df[index]

        # 設置bins的數量
        bins = 30

        # 計算直方圖數據
        counts, bin_edges = np.histogram(data, bins=bins)

        # 根據值的正負設置顏色
        for i in range(len(counts)):
            if bin_edges[i] < 0:
                plt.bar(bin_edges[i], counts[i], width=bin_edges[i+1] - bin_edges[i], color='green', edgecolor='black')
            else:
                plt.bar(bin_edges[i], counts[i], width=bin_edges[i+1] - bin_edges[i], color='red', edgecolor='black')

        plt.xlabel(f'{index} of each trade(%)')
        plt.ylabel('Frequency(Count)')
        plt.title('Histogram of return with Different Colors for Positive and Negative Values')
        plt.show()

        # 計算各區間佔比
        total_count = len(data)
        percentages = {}

        # 計算各區間內數據佔總數據的比值
        thresholds = [-5, -10, -20]
        for threshold in thresholds:
            count = (data <= threshold).sum()
            percentages[threshold] = (count / total_count) * 100

        # 輸出結果
        for threshold, percentage in percentages.items():
            print(f"{threshold}% 占比: {percentage:.2f}%")
    
    def display_topN_cum_return_yearly(self, trades, topN=5):
        """
        Show the topN stocks of each year with the highest cumulative return.

        Parameters
        ----------
        trades : pd.DataFrame
            The trades data.
        topN : int, default 5
            The number of top stocks to show each year.

        Returns
        -------
        None

        Notes
        -----
        This function will show a bar chart where the x-axis is the year and the y-axis is the cumulative return.
        The bars are stacked and the legend shows the stock id.
        The topN stocks of each year are shown in different colors and the other stocks are shown in white with a pattern.
        """
        trades_df = trades.copy()
    
        # 計算累積回報
        trades_df["cum_return"] = (1 + trades_df['return']).groupby(trades_df['stock_id']).cumprod() - 1
        
        # 設定 entry_date 的型態為 datetime
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        
        # 提取年份資訊
        trades_df['year'] = trades_df['entry_date'].dt.year
        trades_df['month'] = trades_df['entry_date'].dt.month
        
        # 按照股票和年份分組計算每年每隻股票的累積回報
        grouped = trades_df.groupby(['year', 'stock_id'])['cum_return'].last().reset_index()

        # 按照股票和月份分組計算每月每隻股票的累積回報
        monthly_grouped = trades_df.groupby(['year', 'month', 'stock_id'])['cum_return'].last().reset_index()
        # monthly_grouped.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/monthly_grouped.csv')

        # 對每年的數據進行處理，選擇前N隻和其他股票的總和
        def process_year_group(group):
            top_stocks = group.nlargest(topN, 'cum_return')
            other_total = group[~group['stock_id'].isin(top_stocks['stock_id'])]['cum_return'].sum()
            other_row = pd.DataFrame({'year': [group['year'].iloc[0]],
                                    'stock_id': ['其他'],
                                    'cum_return': [other_total]})
            return pd.concat([top_stocks, other_row], ignore_index=True)

        # 應用上述函數
        final_data = grouped.groupby('year').apply(process_year_group).reset_index(drop=True)

        # 創建圖表
        fig = go.Figure()

        # 首先添加"其他"類別的條形圖追蹤
        other_data = final_data[final_data['stock_id'] == '其他']
        if not other_data.empty:
            fig.add_trace(go.Bar(
                x=other_data['year'],
                y=other_data['cum_return'],
                name='其他',
                marker=dict(
                    color='white',
                    pattern_shape="/"
                )
            ))

        # 然後添加其他股票的追蹤
        for stock_id in final_data['stock_id'].unique():
            if stock_id != '其他':
                stock_data = final_data[final_data['stock_id'] == stock_id]
                fig.add_trace(go.Bar(
                    x=stock_data['year'],
                    y=stock_data['cum_return'],
                    name=f'Stock {stock_id}'
                ))

        # 更新佈局
        fig.update_layout(
            barmode='stack',
            title=f'年度前{topN}高報酬所佔全部比例',
            xaxis=dict(title='年份'),
            yaxis=dict(title='累積回報'),
            legend_title="公司代號"
        )

        # 顯示圖表
        fig.show()

    def display_annual_plot(self, index="company_count"):
        # 計算每年的績效指標，可帶入指標包含: company_count、cum_returns、portfolio_returns
        plt.plot(self.stock_data.index, self.stock_data[index], marker='o', linestyle='-', linewidth=1)  # 'o' 是點樣式, '-' 是線樣式
        plt.title(f'Plot of {index}')
        plt.xlabel('Time(year)')
        plt.ylabel(index)
        plt.grid(True)
        plt.show()


# 用來安全進行除法的函數。如果分母 d 不等於零，則返回 n / d，否則返回 0。
def safe_division(n, d):
    return n / d if d else 0

# 用來將dataframe按照月份排列
def sort_month(df):
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return df[month_order] 

def deannualize(returns, nperiods):
    """
    Convert return expressed in annual terms on a different basis.
    Args:
        * returns (float, Series, DataFrame): Return(s)
        * nperiods (int): Target basis, typically 252 for daily, 12 for
            monthly, etc.
    """
    return np.power(1 + returns, 1.0 / nperiods) - 1.0

def to_excess_returns(returns, rf, nperiods=None):
    """
    Given a series of returns, it will return the excess returns over rf.
    Args:
        * returns (Series, DataFrame): Returns
        * rf (float, Series): `Risk-Free rate(s) <https://www.investopedia.com/terms/r/risk-freerate.asp>`_ expressed in annualized term or return series
        * nperiods (int): Optional. If provided, will convert rf to different
            frequency using deannualize only if rf is a float
    Returns:
        * excess_returns (Series, DataFrame): Returns - rf
    """
    # if type(rf) is float and nperiods is not None:
    if isinstance(rf, float) and nperiods is not None:

        _rf = deannualize(rf, nperiods)
    else:
        _rf = rf

    return returns - _rf

def rebase(prices, value=100):
    """
    Rebase all series to a given intial value.
    This makes comparing/plotting different series
    together easier.
    Args:
        * prices: Expects a price series
        * value (number): starting value for all series.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.div(prices.iloc[0], axis=1) * value
    return prices / prices.iloc[0] * value
