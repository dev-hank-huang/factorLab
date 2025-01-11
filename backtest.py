import sys
import warnings
import datetime
import numpy as np
import pandas as pd
from typing import Union
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import Union, Dict, List
import report
from get_data import Data
from core.backtest_core import backtest_, get_trade_stocks, get_pos_data, get_return_data
from finlab.core import mae_mfe as maemfe

def warning_resample(resample):

  if '+' not in resample and '-' not in resample:
      return

  if '-' in resample and not resample.split('-')[-1].isdigit():
      return

  if '+' in resample:
      r, o = resample.split('+')
  elif '-' in resample:
      r, o = resample.split('-')

  warnings.warn(f"The argument sim(..., resample = '{resample}') will no longer support after 0.1.37.dev1.\n"
                f"please use sim(..., resample='{r}', offset='{o}d')", DeprecationWarning)

def calc_essential_price(price, dates):

    """
    Calculate the essential price for given dates.

    Parameters
    ----------
    price : pandas.Series
        The price data.
    dates : pandas.DatetimeIndex
        The dates to be calculated.

    Returns
    -------
    pandas.Series
        The essential prices for the given dates.
    """

    dt = min(price.index.values[1:] - price.index.values[:-1])

    indexer = price.index.get_indexer(dates + dt)

    valid_idx = np.where(indexer == -1, np.searchsorted(price.index, dates, side='right'), indexer)
    valid_idx = np.where(valid_idx >= len(price), len(price) - 1, valid_idx)

    return price.iloc[valid_idx]

def arguments(price, high, low, open_, position, resample_dates=None, rolling_dates=None, fast_mode=False):

    resample_dates = price.index if resample_dates is None else resample_dates
    rolling_dates = price.index if rolling_dates is None else rolling_dates
    position = position.astype(float).fillna(0)

    if fast_mode:
        date_index = pd.to_datetime(resample_dates)
        position = position.reindex(date_index, method='ffill')
        price = calc_essential_price(price, date_index)
        high = calc_essential_price(high, date_index)
        low = calc_essential_price(low, date_index)
        open_ = calc_essential_price(open_, date_index)
    
    resample_dates = pd.Series(resample_dates).view(np.int64).values
    rolling_dates = pd.Series(rolling_dates).view(np.int64).values

    return [price.values,
            high.values,
            low.values,
            open_.values,
            price.index.view(np.int64),
            price.columns.astype(str).values,
            position.values,
            position.index.view(np.int64),
            position.columns.astype(str).values,
            resample_dates,
            rolling_dates
            ]

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

def generate_bband_signals(
    self,
    ma_periods: List[int] = [120, 200],
    deviations: List[float] = [1.5, 2.0],
    factor_dict: Dict = None
) -> pd.DataFrame:
    """
    生成布林通道交易信號
    
    Parameters:
    -----------
    ma_periods : List[int]
        移動平均期數列表
    deviations : List[float]
        標準差倍數列表
    factor_dict : Dict
        因子信號字典
    
    Returns:
    --------
    pd.DataFrame : 最終交易信號
    """
    # 獲取價格和成交量數據
    close = self.get('price:close')
    volume = self.get('price:volume')
    
    # 計算成交量確認信號
    vol_ma = volume.rolling(window=20).mean()
    volume_confirm = volume > vol_ma
    
    # 初始化綜合信號矩陣
    final_signals = pd.DataFrame(False, index=close.index, columns=close.columns)
    
    # 計算各組參數的信號
    for ma in ma_periods:
        for dev in deviations:
            # 使用內建的indicator計算布林通道
            upper, middle, lower = self.indicator(
                'BBANDS',
                timeperiod=ma,
                nbdevup=dev,
                nbdevdn=dev,
                resample='D'
            )
            
            # 進場條件：價格突破上軌且成交量確認
            entries = (close > upper) & (close.shift(1) <= upper) & volume_confirm
            
            # 出場條件：跌破中軌或下軌
            exits = (close < lower) | (close < middle)
            
            # 生成持有信號
            signals = entries.copy()
            holding = False
            
            for i in range(1, len(close)):
                if not holding and entries.iloc[i].any():
                    signals.iloc[i] = True
                    holding = True
                elif holding and exits.iloc[i].any():
                    signals.iloc[i] = False
                    holding = False
                elif holding:
                    signals.iloc[i] = True
                    
            final_signals = final_signals | signals
    
    # 如果有因子信號，進行結合
    if factor_dict is not None:
        for factor_name, factor_signal in factor_dict.items():
            final_signals = final_signals & factor_signal
    
    return final_signals

def enhanced_sim(
    self,
    ma_periods: List[int] = [120, 200],
    deviations: List[float] = [1.5, 2.0],
    factor_dict: Dict = None,
    **sim_params
):
    """
    整合布林通道策略的強化版sim函數
    
    Parameters:
    -----------
    data_api : 數據API物件
    ma_periods : List[int]
        移動平均期數列表
    deviations : List[float]
        標準差倍數列表
    factor_dict : Dict
        因子信號字典
    sim_params : dict
        原sim函數的其他參數
    
    Returns:
    --------
    Report : 回測報告物件
    """
    
    # 生成交易信號
    position = self.generate_bband_signals(
        ma_periods=ma_periods,
        deviations=deviations,
        factor_dict=factor_dict
    )
    
    # 使用原始sim函數進行回測
    return sim(
        position=position,
        **sim_params
    )


def sim(position: Union[pd.DataFrame, pd.Series],
        resample:Union[str, None]=None, resample_offset:Union[str, None] = None,
        position_limit:float=1, fee_ratio:float=1.425/1000,
        tax_ratio: float=3/1000, stop_loss: Union[float, None]=None,
        
        # 跟rooling相關參數
        rolling_ratio:float=1.0, rolling_freq:Union[str, None]=None,
        rolling_take_profit: Union[float, None]=None, rolling_stop_loss:Union[float, None]=None, 
        profit_rolling_ratio:float=1.0, loss_rolling_ratio:float=1.0,

        take_profit: Union[float, None]=None, trail_stop: Union[float, None]=None, touched_exit: bool=False,
        retain_cost_when_rebalance: bool=False, stop_trading_next_period: bool=True, live_performance_start:Union[str, None]=None,
        mae_mfe_window:int=0, mae_mfe_window_step:int=1, fast_mode=False, data=None):


     # check type of position
    """
    Parameters
    ----------
    position : pd.DataFrame
        Position matrix where index is the date and column is the stock id.
        The value is the position percentage of each stock.
    resample : str, default None
        The resample frequency. If None, no resample.
        e.g. '1d' for daily resample.
    resample_offset : str, default None
        The offset for resample frequency. If None, no offset.
        e.g. '+1' for resample with 1 day offset.
    position_limit : float, default 1
        The maximum position percentage of each stock.
    fee_ratio : float, default 1.425 / 1000
        The trading fee ratio.
    tax_ratio : float, default 3 / 1000
        The tax ratio.
    stop_loss : float, default None
        The stop loss percentage.
    rolling_ratio : float, default 1.0
        The rolling ratio for the rolling window.
    rolling_freq : str, default None
        The rolling frequency. If None, no rolling.
        e.g. '1d' for daily rolling.
    rolling_take_profit : float, default None
        The take profit percentage for rolling.
    rolling_stop_loss : float, default None
        The stop loss percentage for rolling.
    profit_rolling_ratio : float, default 1.0
        The rolling ratio for the profit window.
    loss_rolling_ratio : float, default 1.0
        The rolling ratio for the loss window.
    take_profit : float, default None
        The take profit percentage.
    trail_stop : float, default None
        The trail stop percentage.
    touched_exit : bool, default False
        Whether to consider the exit date when calculating the profit.
    retain_cost_when_rebalance : bool, default False
        Whether to retain the cost when rebalancing.
    stop_trading_next_period : bool, default True
        Whether to stop trading in the next period after the end of the backtest.
    live_performance_start : str, default None
        The start date of the live performance period.
        e.g. '2021-01-01' for starting from 2021-01-01.
    mae_mfe_window : int, default 0
        The window size for the mae mfe calculation.
    mae_mfe_window_step : int, default 1
        The window step for the mae mfe calculation.
    fast_mode : bool, default False
        Whether to use the fast mode for backtesting.
    data : Data, default None
        The data object.

    Returns
    -------
    Report
        The report object containing the backtest results.

    Notes
    -----
    The backtest function will return a Report object containing the backtest results.
    The Report object contains the following attributes:

    - cum_returns : pd.Series
        The cumulative returns of the strategy.
    - portfolio_returns : pd.Series
        The daily returns of the strategy.
    - mae_mfe : pd.DataFrame
        The mae mfe metrics for each trade.
    - trades : pd.DataFrame
        The trades dataframe containing the entry and exit dates, positions, and returns.
    - next_trading_date : str
        The next trading date after the end of the backtest.
    - current_trades : pd.DataFrame
        The current trades dataframe containing the entry and exit dates, positions, and returns.
    - next_weights : pd.Series
        The next weights for each stock.
    - weights : pd.Series
        The current weights for each stock.
    """
    if not isinstance(position.index, pd.DatetimeIndex):
        raise TypeError("Expected the dataframe to have a DatetimeIndex")
    
    #########
    #測試用的#
    #########
    # price = pd.read_csv('../Data/verify_rolling/stock_price.csv').set_index('date').astype('float64')


    if isinstance(data, Data):
        price = data.get('price:close')
    else:
        data=Data()
        price = data.get('price:close')

    high = price
    low = price
    open_ = price
    if touched_exit:
        high = data.get('price:high').reindex_like(price)
        low =data.get('price:low').reindex_like(price)
        open_ = data.get('price:open').reindex_like(price) 

    if not isinstance(price.index[0], pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        open_.index = pd.to_datetime(open_.index)

    assert len(position.shape) >= 2
    delta_time_rebalance = position.index[-1] - position.index[-3]
    backtest_to_end = position.index[-1] + \
        delta_time_rebalance > price.index[-1]

    tz = position.index.tz
    now = datetime.datetime.now(tz=tz)

    position = position[(position.index <= price.index[-1]) | (position.index <= now)]
    backtest_end_date = price.index[-1] if backtest_to_end else position.index[-1]

    # resample dates
    dates = None
    next_trading_date = position.index[-1]
    if isinstance(resample, str):

        warning_resample(resample)

        # add additional day offset
        offset_days = 0
        if '+' in resample:
            offset_days = int(resample.split('+')[-1])
            resample = resample.split('+')[0]
        if '-' in resample and resample.split('-')[-1].isdigit():
            offset_days = -int(resample.split('-')[-1])
            resample = resample.split('-')[0]

        # generate rebalance dates
        alldates = pd.date_range(
            position.index[0], 
            position.index[-1] + datetime.timedelta(days=720), 
            freq=resample, tz=tz)

        alldates += DateOffset(days=offset_days)

        if resample_offset is not None:
            alldates += to_offset(resample_offset)

        dates = [d for d in alldates if position.index[0]
                 <= d and d <= position.index[-1]]

        # calculate the latest trading date
        next_trading_date = min(
           set(alldates) - set(dates))

        if dates[-1] != position.index[-1]:
            dates += [next_trading_date]

    if rolling_freq is None and resample is not None:
        rolling_freq = resample

    # rolling dates
    rolling_dates = None
    next_rolling_date = position.index[-1]
    if isinstance(rolling_freq, str):

        warning_resample(rolling_freq)

        # add additional day offset
        offset_days = 0
        if '+' in rolling_freq:
            offset_days = int(rolling_freq.split('+')[-1])
            rolling_freq = rolling_freq.split('+')[0]
        if '-' in rolling_freq and rolling_freq.split('-')[-1].isdigit():
            offset_days = -int(resample.split('-')[-1])
            rolling_freq = rolling_freq.split('-')[0]

        # generate rebalance dates
        alldates = pd.date_range(
            position.index[0], 
            position.index[-1] + datetime.timedelta(days=720), 
            freq=rolling_freq, tz=tz)
        
        rolling_dates = [d for d in alldates if position.index[0]
                 <= d and d <= position.index[-1]]

        # calculate the latest trading date
        next_rolling_date = min(
           set(alldates) - set(rolling_dates))

        if rolling_dates[-1] != position.index[-1]:
            rolling_dates += [next_rolling_date]

    if rolling_take_profit is None or rolling_take_profit == 0:
        rolling_take_profit = np.inf

    if rolling_stop_loss is None or rolling_stop_loss == 0:
        rolling_stop_loss = -np.inf

    if stop_loss is None or stop_loss == 0:
        stop_loss = 1

    if take_profit is None or take_profit == 0:
        take_profit = np.inf

    if trail_stop is None or trail_stop == 0:
        trail_stop = np.inf

    if dates is not None:
        position = position.reindex(dates, method='ffill')

    # 準備回測參數
    args = arguments(price, high, low, open_, position, dates, rolling_dates, fast_mode=fast_mode)

    ####################################
    args.append(position.columns.values)  # 新增股票代號 (for cashflow)

    # pd.DataFrame(args).to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/input_args.csv')
    ####################################

    creturn_value = backtest_(*args,
                              fee_ratio=fee_ratio, tax_ratio=tax_ratio, 
                              rolling_ratio=rolling_ratio,profit_rolling_ratio=profit_rolling_ratio,
                              loss_rolling_ratio=loss_rolling_ratio,rolling_take_profit=rolling_take_profit,
                              rolling_stop_loss=rolling_stop_loss,
                              stop_loss=stop_loss, take_profit=take_profit, trail_stop=trail_stop,
                              touched_exit=touched_exit, position_limit=position_limit,
                              retain_cost_when_rebalance=retain_cost_when_rebalance,
                              stop_trading_next_period=stop_trading_next_period,
                              mae_mfe_window=mae_mfe_window, mae_mfe_window_step=mae_mfe_window_step)
    # # test
    # creturn_value_df = pd.DataFrame(creturn_value, index=price.index, columns=['creturn_value'])
    # creturn_value_df.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/2024_code/exp_data/creturn_value.csv')

    ####################################
    # Retrieve pos data and convert to DataFrame
    # pos_data = get_pos_data()
    # print(f"Number of items in return_data: {len(pos_data)}")
    # if return_data:
    #     print(f"Sample data: {pos_data[0]}")
    # cashflow_df = pd.DataFrame([pos for _, pos in pos_data], 
    #                            index=pd.to_datetime([date for date, _ in pos_data]))
    # print(cashflow_df.head)
    # # sort the df by column index (stock id)
    # cashflow_df = cashflow_df.sort_index(axis=1).rename_axis('date').rename_axis('company_symbol', axis=1)

    # cashflow_df.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/cashflow.csv')
    ####################################

    ####################################
    # return_data = get_return_data()
    # print(f"Number of items in return_data: {len(return_data)}")
    # if return_data:
    #     print(f"Sample data: {return_data[0]}")
    # return_df = pd.DataFrame([ret for _, ret in return_data], 
    #                         index=pd.to_datetime([date for date, _ in return_data]))
    # return_df = return_df.sort_index(axis=1).rename_axis('date').rename_axis('company_symbol', axis=1)
    # return_df.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/daily_return.csv')
    ####################################
    
    
    total_weight = position.abs().sum(axis=1)

    position = position.div(total_weight.where(total_weight!=0, np.nan), axis=0).fillna(0)\
                       .clip(-abs(position_limit), abs(position_limit))
    
    creturn_dates = dates if dates and fast_mode else price.index

    creturn = (pd.Series(creturn_value, creturn_dates)
                # remove the begining of creturn since there is no pct change
                .pipe(lambda df: df[(df != 1).cumsum().shift(-1, fill_value=1) != 0])
                # remove the tail of creturn for verification
                .loc[:backtest_end_date]
                # replace creturn to 1 if creturn is None
                .pipe(lambda df: df if len(df) != 0 else pd.Series(1, position.index)))
    
    # test
    # 將 Series 轉換為 DataFrame
    creturn_df = creturn.to_frame(name='creturn_value')
    # 添加日期列（如果索引不是日期，請相應調整）
    creturn_df.reset_index(inplace=True)
    creturn_df.columns = ['creturn_dates', 'creturn_value']
    # creturn_df.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/creturn.csv', index=False)
    
    position = position.loc[creturn.index[0]:]

    price_index = args[4]
    position_columns = args[8]
    trades, operation_and_weight = get_trade_stocks(position_columns, 
                                                    price_index, touched_exit=touched_exit)
    # test
    # trades.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/processed_trades.csv')

    ####################################
    # refine mae mfe dataframe
    ####################################
    def refine_mae_mfe():
        if len(maemfe.mae_mfe) == 0:
            return pd.DataFrame()

        m = pd.DataFrame(maemfe.mae_mfe)
        # m.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/mae_mfe_start.csv')
        nsets = int((m.shape[1]-1) / 6)

        metrics = ['mae', 'gmfe', 'bmfe', 'mdd', 'pdays', 'return']

        tuples = sum([[(n, metric) if n == 'exit' else (n * mae_mfe_window_step, metric)
                       for metric in metrics] for n in list(range(nsets)) + ['exit']], [])

        m.columns = pd.MultiIndex.from_tuples(
            tuples, names=["window", "metric"])
        m.index.name = 'trade_index'
        m[m == -1] = np.nan

        exit = m.exit.copy()

        if touched_exit and len(m) > 0 and 'exit' in m.columns:
            m['exit'] = (exit
                .assign(gmfe=exit.gmfe.clip(-abs(stop_loss), abs(take_profit)))
                .assign(bmfe=exit.bmfe.clip(-abs(stop_loss), abs(take_profit)))
                .assign(mae=exit.mae.clip(-abs(stop_loss), abs(take_profit)))
                .assign(mdd=exit.mdd.clip(-abs(stop_loss), abs(take_profit))))
        # m.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/mae_mfe_end.csv')
        return m
    
    m = refine_mae_mfe()

    # test
    # m.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/mae_mfe_data.csv')

    #####################################
    # refine trades dataframe
    ####################################
    def convert_datetime_series(df):
        cols = ['entry_date', 'exit_date', 'entry_sig_date', 'exit_sig_date']
        df[cols] = df[cols].apply(lambda s: pd.to_datetime(s).dt.tz_localize(tz))
        return df

    def assign_exit_nat(df):
        cols = ['exit_date', 'exit_sig_date']
        df[cols] = df[cols].loc[df.exit_index != -1]
        return df

    trades = (pd.DataFrame(trades, 
                           columns=['stock_id', 'entry_date', 'exit_date',
                                    'entry_sig_date', 'exit_sig_date', 'position', 
                                    'period', 'entry_index', 'exit_index'])
              .rename_axis('trade_index')
              .pipe(convert_datetime_series)
              .pipe(assign_exit_nat)
              )
    
    if len(trades) != 0:
        trades = trades.assign(**{'return': m.iloc[:, -1] - 2*fee_ratio - tax_ratio})

    if touched_exit:
        trades['return'] = trades['return'].clip(-abs(stop_loss), abs(take_profit))

    # trades = trades.drop(['entry_index', 'exit_index'], axis=1)

    
    daily_creturn = rebase(creturn.resample('1d').last().dropna().ffill())
    
    stock_data = pd.DataFrame(index = creturn.index)
    stock_data['portfolio_returns'] = daily_creturn
    stock_data['cum_returns'] = creturn
    stock_data['company_count'] = (position != 0).sum(axis=1)

    r = report.Report(stock_data, position, data)
    r.mae_mfe = m
    r.trades = trades

    # r.portfolio_returns.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/portfolio_returns.csv')
    # r.position.to_csv('C:/Users/Hank/Downloads/stock_factor_lab_202408/stock_factor_lab/output_file/final_positions.csv')
    # r.trades.to_csv('C:/Users/Hank/stock_factor_lab_202408/stock_factor_lab/output_file/final_trades.csv')

    ####################################
    # Add the cashflow attribute to the Report object
    # r.cashflow = cashflow_df
    # r.daily_return = return_df
    ####################################
    
    # calculate weights
    if len(operation_and_weight['weights']) != 0:
        r.weights = pd.Series(operation_and_weight['weights'])
        r.weights.index = r.position.columns[r.weights.index]
    else:
        r.weights = pd.Series(dtype='float64')


    # calculate next weights
    if len(operation_and_weight['next_weights']) != 0:
        r.next_weights = pd.Series(operation_and_weight['next_weights'])
        r.next_weights.index = r.position.columns[r.next_weights.index]
    else:
        r.next_weights = pd.Series(dtype='float64')


    # calculate actions    
    if len(operation_and_weight['actions']) != 0:
        # find selling and buying stocks
        r.actions = pd.Series(operation_and_weight['actions'])
        r.actions.index = r.position.columns[r.actions.index]
    else:
        r.actions = pd.Series(dtype=object)

    if len(r.actions) != 0:

        actions = r.actions

        sell_sids = actions[actions == 'exit'].index
        sell_instant_sids = actions[(actions == 'sl') | (actions == 'tp')].index
        buy_sids = actions[actions == 'enter'].index

        if len(trades):
            # check if the sell stocks are in the current position
            # assert len(set(sell_sids) - set(trades.stock_id[trades.exit_sig_date.isnull()])) == 0

            # fill exit_sig_date and exit_date
            temp = trades.loc[trades.stock_id.isin(sell_sids), 'exit_sig_date'].fillna(r.position.index[-1])
            trades.loc[trades.stock_id.isin(sell_sids), 'exit_sig_date'] = temp

            temp = trades.loc[trades.stock_id.isin(sell_instant_sids), 'exit_sig_date'].fillna(price.index[-1])
            trades.loc[trades.stock_id.isin(sell_instant_sids), 'exit_sig_date'] = temp

            r.trades = pd.concat([r.trades, pd.DataFrame({
              'stock_id': buy_sids,
              'entry_date': pd.NaT,
              'entry_sig_date': r.position.index[-1],
              'exit_date': pd.NaT,
              'exit_sig_date': pd.NaT,
            })], ignore_index=True)

            r.trades['exit_sig_date'] = pd.to_datetime(r.trades.exit_sig_date)

    if len(trades) != 0:
        trades = r.trades
        mae_mfe = r.mae_mfe
        exit_mae_mfe = mae_mfe['exit'].copy()
        exit_mae_mfe = exit_mae_mfe.drop(columns=['return'])
        r.trades = pd.concat([trades, exit_mae_mfe], axis=1)
        r.trades.index.name = 'trade_index'

        # calculate r.current_trades
        # find trade without end or end today
        maxday = max(r.trades.entry_sig_date.max(), r.trades.exit_sig_date.max())
        latest_entry_day = r.trades.entry_sig_date[r.trades.entry_date.notna()].max()
        r.current_trades = r.trades[
                (r.trades.entry_sig_date == maxday )
                | (r.trades.exit_sig_date == maxday)
                | (r.trades.exit_sig_date > latest_entry_day)
                | (r.trades.entry_sig_date == latest_entry_day)
                | (r.trades.exit_sig_date.isnull())
                ].set_index('stock_id')

        r.next_trading_date = max(r.current_trades.entry_sig_date.max(), r.current_trades.exit_sig_date.max())

        r.current_trades['weight'] = 0
        if len(r.weights) != 0:
            r.current_trades['weight'] = r.weights.reindex(r.current_trades.index).fillna(0)

        r.current_trades['next_weights'] = 0
        if len(r.next_weights) != 0:
            r.current_trades['next_weights'] = r.next_weights.reindex(r.current_trades.index).fillna(0)
    
    r.trades = r.trades.drop(['entry_index', 'exit_index'], axis=1)

    return r
