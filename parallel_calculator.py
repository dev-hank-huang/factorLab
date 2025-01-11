from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm.asyncio import tqdm
import asyncio
import logging
import psutil
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
from factors.registry import FactorRegistry
from shareDataManager import SharedDataManager
from trading_strategies.registry import StrategyRegistry

class ResourceManager:
    """資源管理類，處理系統資源監控和優化"""
    MIN_MEMORY_PER_PROCESS = 2 * 1024 * 1024 * 1024

    @staticmethod
    def get_optimal_process_count(n_processes: int = None) -> int:
        available_memory = psutil.virtual_memory().available
        return min(
            n_processes or (cpu_count() - 2),
            int(available_memory / ResourceManager.MIN_MEMORY_PER_PROCESS)
        )

    @staticmethod
    async def monitor_resources(logger):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.Process().memory_percent()
        
        if cpu_usage > 90 or memory_usage > 90:
            logger.warning(f"資源使用過高: CPU {cpu_usage}%, Memory {memory_usage}%")
            await asyncio.sleep(1)

class ProgressTracker:
    """進度追蹤類，管理計算進度和時間估算"""
    def __init__(self, logger):
        self.progress = {}
        self.start_times = {}
        self.logger = logger
        self.total_factors = 0
        self.completed_factors = 0

    def start_factor(self, factor_name: str):
        current_time = datetime.now()
        self.start_times[factor_name] = current_time
        self.progress[factor_name] = {
            'status': 'processing',
            'start_time': current_time,
            'attempts': 0
        }

    def complete_factor(self, factor_name: str, cache_hit: bool = False, success: bool = True):
        """
        記錄因子計算完成狀態
        
        Parameters:
        -----------
        factor_name : str
            因子名稱
        cache_hit : bool, optional
            是否命中快取，默認False
        success : bool, optional
            計算是否成功，默認True
        """
        duration = (datetime.now() - self.start_times[factor_name]).total_seconds()
        status = 'completed' if success else 'failed'
        
        self.progress[factor_name].update({
            'status': status,
            'duration': duration,
            'cache_hit': cache_hit
        })
        
        if success:
            self.completed_factors += 1
            
        self.logger.info(
            f"因子 {factor_name} {status}，"
            f"耗時：{duration:.2f} 秒"
            f"{'(快取命中)' if cache_hit else ''}"
        )

class HybridParallelCalculator:
    """混合並行計算器主類"""
    def __init__(self, data, n_processes=None, n_threads=None):
        self.data = data
        self.n_processes = n_processes
        self.n_threads = n_threads or min(32, cpu_count())
        self.logger = self._initialize_logger()
        self.progress_tracker = ProgressTracker(self.logger)
        self.resource_manager = ResourceManager()
        self._strategy_registry = StrategyRegistry()

    def _initialize_logger(self) -> logging.Logger:
        logger = logging.getLogger('ParallelFactorCalculator')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 添加控制台處理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 添加文件處理器
            file_handler = logging.FileHandler(
                'logs/factor_calculation.log',
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    async def calculate_factors(self, 
                              factor_names: List[str],
                              use_cache: bool = True,
                              force_refresh: bool = False,
                              max_age_minutes: int = 480,
                              strategy_configs: dict = None) -> Dict:
        """主要計算方法"""
        start_time = time.time()
        self.logger.info(f"開始計算 {len(factor_names)} 個因子")
        
        shared_manager = SharedDataManager()
        shared_data = shared_manager.initialize_shared_data(self.data)
        
        try:
            # 執行計算流程
            results = await self._execute_calculation_pipeline(
                factor_names,
                shared_data,
                use_cache,
                force_refresh,
                max_age_minutes,
                strategy_configs
            )
            
            # 報告執行統計
            self._report_statistics(start_time, len(factor_names), results)
            return results
            
        finally:
            shared_manager.cleanup()

    async def _execute_calculation_pipeline(self,
                                         factor_names: List[str],
                                         shared_data: Dict,
                                         use_cache: bool,
                                         force_refresh: bool,
                                         max_age_minutes: int,
                                         strategy_configs: Dict = None) -> Dict:
        """執行計算pipeline"""
        
        # 檢查快取
        cache_results, to_calculate = await self._check_cache(
            factor_names,
            use_cache,
            force_refresh,
            max_age_minutes
        )
        
        # 計算新因子
        if to_calculate:
            new_results = await self._calculate_new_factors(
                to_calculate,
                shared_data
            )
            cache_results.update(new_results)
        
        # 第三步：如果有策略配置，計算策略信號並整合
        if strategy_configs:
            try:
                strategy_signals = await self._calculate_strategy_signals(
                    strategy_configs=strategy_configs,
                    use_cache=use_cache,
                    force_refresh=force_refresh
                )
                
                return self._combine_factor_and_strategy_signals(
                    factor_signals=cache_results,
                    strategy_signals=strategy_signals,
                    combination_type=strategy_configs.get('combination_type', 'AND')
                )
                
            except Exception as e:
                self.logger.error(f"策略信號計算錯誤: {str(e)}")
                return cache_results
        
        return cache_results

    async def _check_cache(self, factor_names, use_cache, force_refresh, max_age_minutes):
        cache_results = {}
        factors_to_calculate = []
        
        for factor_name in factor_names:
            try:
                cache_key = f"{factor_name}_cache"
                if use_cache and not force_refresh:
                    cached_result = self.data.memory_cache.get(cache_key, max_age_minutes)
                    if cached_result is not None:
                        cache_results[factor_name] = cached_result
                        self.progress_tracker.complete_factor(factor_name, cache_hit=True)
                        continue
                factors_to_calculate.append(factor_name)
            except Exception as e:
                self.logger.error(f"快取檢查錯誤 {factor_name}: {str(e)}")
                factors_to_calculate.append(factor_name)
        
        return cache_results, factors_to_calculate

    async def _calculate_new_factors(self, factors_to_calculate, shared_data_dict):
        """計算新因子的實作"""
        calculation_results = {}
        
        if not factors_to_calculate:
            self.logger.warning("沒有需要計算的因子")
            return calculation_results
            
        try:
            n_processes = self.resource_manager.get_optimal_process_count(len(factors_to_calculate))
            self.logger.info(f"使用 {n_processes} 個進程進行因子計算")
            
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = []
                for factor_name in factors_to_calculate:
                    self.progress_tracker.start_factor(factor_name)
                    future = executor.submit(
                        self._calculate_single_factor,
                        (factor_name, shared_data_dict)
                    )
                    futures.append(future)
                
                for future in tqdm(futures, desc="因子計算進度"):
                    try:
                        name, result = future.result(timeout=1200)
                        if result is not None:
                            calculation_results[name] = result
                            self.progress_tracker.complete_factor(
                                factor_name=name,
                                cache_hit=False,
                                success=True
                            )
                        else:
                            self.logger.warning(f"因子 {name} 計算結果為空")
                            self.progress_tracker.complete_factor(
                                factor_name=name,
                                cache_hit=False,
                                success=False
                            )
                    except Exception as e:
                        self.logger.error(f"處理因子結果時發生錯誤: {str(e)}")
                        continue
                    
                    await self.resource_manager.monitor_resources(self.logger)
            
            return calculation_results
            
        except Exception as e:
            self.logger.error(f"計算新因子時發生錯誤: {str(e)}")
            return calculation_results

    def _calculate_single_factor(self, args: Tuple):
        factor_name, shared_data_dict = args
        try:
            shared_manager = SharedDataManager()
            price_data = shared_manager.get_shared_data(shared_data_dict)
            
            calculator = FactorRegistry.get_calculator(factor_name)
            if calculator is None:
                return factor_name, None
                
            calculator_instance = calculator()
            result = calculator_instance.calculate_with_shared_data(price_data)
            
            if result is not None and not result.empty:
                # 將結果轉換為 numpy array
                factor_slice = result.divide_slice(quantile=4, ascending=False)['Quantile_1']
                return factor_name, factor_slice.to_numpy()
                
            return factor_name, None
            
        except Exception as e:
            self.logger.error(f"因子 {factor_name} 計算時發生錯誤: {str(e)}")
            return factor_name, None
        finally:
            if 'shared_manager' in locals():
                shared_manager.cleanup()
    
    def _combine_factor_and_strategy_signals(self,
                                       factor_signals: Dict[str, np.ndarray],
                                       strategy_signals: Dict[str, pd.DataFrame],
                                       combination_type: str = "AND") -> Dict[str, pd.DataFrame]:
        """組合因子和策略信號，確保每個因子與每組參數都進行組合"""
        try:
            combined_results = {}
            index = self.data.get('price:close').index
            columns = self.data.get('price:close').columns
            
            for factor_name, factor_signal in factor_signals.items():
                factor_df = pd.DataFrame(factor_signal, index=index, columns=columns) if isinstance(factor_signal, np.ndarray) else factor_signal
                
                for strategy_name, strategy_signal in strategy_signals.items():
                    combined_name = f"{factor_name}_{strategy_name}"
                    
                    if combination_type == "AND":
                        combined = factor_df & strategy_signal
                    else:
                        combined = factor_df | strategy_signal
                    
                    signal_count = combined.sum().sum()
                    print(f"{combined_name} 組合後信號數量: {signal_count}")
                    combined_results[combined_name] = combined
            
            return combined_results
                
        except Exception as e:
            self.logger.error(f"組合信號時發生錯誤: {str(e)}")
            return factor_signals
        
    async def _calculate_strategy_signals(self,
                                    strategy_configs: Dict[str, Any],
                                    use_cache: bool = True,
                                    force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        signals = {}
        
        try:
            strategy_list = strategy_configs.get('strategies', [])
            if not strategy_list:
                self.logger.warning("策略列表為空")
                return signals
            
            for strategy_config in strategy_list:
                try:
                    strategy_name = strategy_config['name']
                    params_list = strategy_config.get('params_list', [])
                    
                    # 根據策略類型選擇正確的參數識別邏輯
                    for params in params_list:
                        try:
                            param_identifier = self._generate_strategy_identifier(strategy_name, params)
                            unique_strategy_name = f"{param_identifier}" # {strategy_name}_
                            
                            # print(f"\n處理策略: {unique_strategy_name}")
                            # print(f"使用參數: {params}")
                            
                            # 獲取對應的策略類
                            strategy_class = self._strategy_registry.get_strategy(strategy_name)
                            if strategy_class is None:
                                self.logger.warning(f"找不到策略 {strategy_name}")
                                continue
                                
                            strategy = strategy_class()
                            signal = strategy.generate_signals(self.data, **params)
                            
                            if signal is not None and not signal.empty:
                                signals[unique_strategy_name] = signal
                                signal_count = signal.sum().sum()
                                print(f"策略 {unique_strategy_name} 生成了 {signal_count} 個信號")
                            
                        except Exception as e:
                            self.logger.error(f"處理策略 {strategy_name} 的參數時發生錯誤: {str(e)}")
                            continue
                            
                except Exception as e:
                    self.logger.error(f"處理策略配置時發生錯誤: {str(e)}")
                    continue
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"計算策略信號時發生錯誤: {str(e)}")
            return signals

    def _generate_strategy_identifier(self, strategy_name: str, params: Dict) -> str:
        """根據策略類型生成參數識別名稱"""
        try:
            if strategy_name == 'BOLLINGER_BREAKTHROUGH':
                return f"BB_T{params['timeperiod']}_D{str(params['nbdevup']).replace('.', 'p')}"
                
            elif strategy_name == 'VOLUME_MA_BREAKTHROUGH':
                return f"VMB_M{params['ma_period']}"

            elif strategy_name == 'ENHANCED_BBAND':
                return f"ENHANCED_B{params['base_period']}_P{params['profit_target']}_S{params['stop_loss']}"
            
            else:
                return f"PARAM_{hash(str(params))}"
            
        except KeyError as e:
            self.logger.error(f"生成策略識別符時缺少必要參數: {str(e)}")
            return f"ERROR_{hash(str(params))}"
        
    def _report_statistics(self, start_time: float, total_factors: int, results: Dict):
        """
        生成並記錄因子計算的詳細統計報告。
        
        參數:
            start_time: 計算開始的時間戳
            total_factors: 要計算的因子總數
            results: 計算結果字典
        """
        # 計算基本統計
        total_time = time.time() - start_time
        cache_hits = sum(1 for status in self.progress_tracker.progress.values() 
                        if status.get('cache_hit', False))
        cache_hit_rate = (cache_hits / total_factors) * 100
        
        # 計算成功率
        successful_calculations = len(results)
        success_rate = (successful_calculations / total_factors) * 100
        
        # 計算平均處理時間
        calculation_times = [
            status['duration'] 
            for status in self.progress_tracker.progress.values() 
            if 'duration' in status and not status.get('cache_hit', False)
        ]
        avg_calculation_time = (
            sum(calculation_times) / len(calculation_times) 
            if calculation_times else 0
        )

        # 記錄詳細統計報告
        self.logger.info(
            f"\n因子計算統計報告:"
            f"\n-------------------"
            f"\n總計算時間: {total_time:.2f} 秒"
            f"\n處理因子數: {total_factors}"
            f"\n成功完成數: {successful_calculations}"
            f"\n成功率: {success_rate:.1f}%"
            f"\n快取命中數: {cache_hits}"
            f"\n快取命中率: {cache_hit_rate:.1f}%"
            f"\n平均計算時間: {avg_calculation_time:.2f} 秒/因子"
            f"\n-------------------"
        )

        # 如果有計算失敗的情況，記錄詳細信息
        failed_factors = total_factors - successful_calculations
        if failed_factors > 0:
            self.logger.warning(
                f"有 {failed_factors} 個因子計算失敗，"
                f"請檢查日誌了解詳細信息。"
            )
            
            
        
# Fixing
async def _integrate_bband_signals(self, results, bband_params):
    """布林通道信號整合實作"""
    self.logger.info("開始計算布林通道信號")
    
    try:
        loop = asyncio.get_event_loop()
        bband_signals = await loop.run_in_executor(
            None,
            self._calculate_bband_signals,
            bband_params
        )
        
        if bband_signals is not None:
            date_index = self.data.get('price:close').index
            columns = self.data.get('price:close').columns
            
            combined_results = {}
            for name, value in results.items():
                try:
                    # 確保 bband_signals 是 DataFrame 格式
                    if isinstance(bband_signals, np.ndarray):
                        bband_df = pd.DataFrame(bband_signals, index=date_index, columns=columns)
                    else:
                        bband_df = bband_signals
                    
                    # 確保 value 是 DataFrame 格式
                    if isinstance(value, np.ndarray):
                        value_df = pd.DataFrame(value, index=date_index, columns=columns)
                    else:
                        value_df = value
                    
                    # 執行布林運算並保存結果
                    result = self._safe_boolean_operation(value_df, bband_df)
                    if result is not None:
                        if isinstance(result, np.ndarray):
                            result = pd.DataFrame(result, index=date_index, columns=columns)
                        combined_results[f"{name}_bband"] = result
                
                except Exception as e:
                    self.logger.error(f"處理因子 {name} 時發生錯誤: {str(e)}")
                    continue
            
            return combined_results if combined_results else results
            
        return results
    
    except Exception as e:
        self.logger.error(f"整合布林通道信號時發生錯誤: {str(e)}")
        return results

def _safe_boolean_operation(self, value1, value2):
    """安全地執行布林運算並確保返回 DataFrame"""
    try:
        # 確保輸入都是 DataFrame
        if isinstance(value1, pd.DataFrame) and isinstance(value2, pd.DataFrame):
            return value1 & value2
        else:
            # 這裡不需要轉換為 numpy array，保持 DataFrame 格式
            return None
            
    except Exception as e:
        self.logger.error(f"布林運算時發生錯誤: {str(e)}")
        return None

def _calculate_bband_signals(self, bband_params):
        """計算布林通道信號並返回 DataFrame"""
        try:
            cache_key = f"bband_signals_{hash(str(bband_params))}"
            cached_signals = self.data.memory_cache.get(cache_key, 60)
            if cached_signals is not None:
                return cached_signals

            close = self.data.get('price:close')
            volume = self.data.get('price:volume')
            
            bands = self.data.indicator('BBANDS', **bband_params)
            vol_ma = volume.rolling(window=20).mean()
            volume_confirm = volume > vol_ma

            entries = (close > bands[0]) & (close.shift(1) <= bands[0]) & volume_confirm
            exits = (close < bands[2]) | (close < bands[1])
            signals = entries.hold_until(exits)
            
            # 直接返回 DataFrame
            return signals

        except Exception as e:
            self.logger.error(f"布林通道信號計算錯誤: {str(e)}")
            return None
        