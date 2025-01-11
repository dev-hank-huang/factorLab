from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('LFHP')
class LFHPCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required price data
        close = data.get("price:close")
        volume = data.get("price:volume")
        
        # Ensure datetime index
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            
        dfs = {}
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate frequency components
            def calculate_frequency_components(series, window=20):
                components = []
                for i in range(len(series) - window + 1):
                    if i >= window:
                        segment = series.iloc[i-window:i]
                        fft = np.fft.fft(segment)
                        components.append(np.abs(fft[:window//2]))
                return np.array(components)
            
            # Calculate Liquidity Wave Spectrum (LWS)
            price_changes = price.pct_change()
            volume_changes = vol.pct_change()
            
            price_freq = pd.Series(index=price.index, dtype=float)
            volume_freq = pd.Series(index=price.index, dtype=float)
            
            # Rolling frequency analysis
            window = 20
            for i in range(window, len(price)):
                p_segment = price_changes.iloc[i-window:i]
                v_segment = volume_changes.iloc[i-window:i]
                
                p_fft = np.fft.fft(p_segment)
                v_fft = np.fft.fft(v_segment)
                
                # Calculate phase correlation
                p_phase = np.angle(p_fft)
                v_phase = np.angle(v_fft)
                phase_diff = np.exp(1j * (p_phase - v_phase))
                
                price_freq.iloc[i] = np.mean(np.abs(p_fft))
                volume_freq.iloc[i] = np.mean(np.abs(phase_diff))
            
            lws = price_freq * volume_freq
            
            # Calculate Harmonic Resonance Index (HRI)
            def calculate_wavelet_correlation(x, y, window=10):
                x_std = x.rolling(window=window).std()
                y_std = y.rolling(window=window).std()
                covariance = (x * y).rolling(window=window).mean() - \
                            x.rolling(window=window).mean() * y.rolling(window=window).mean()
                correlation = covariance / (x_std * y_std)
                return correlation
            
            price_wavelet = price_changes.rolling(window=10).apply(
                lambda x: np.std(x) * np.sign(x.mean())
            )
            volume_wavelet = volume_changes.rolling(window=10).apply(
                lambda x: np.std(x) * np.sign(x.mean())
            )
            
            hri = calculate_wavelet_correlation(price_wavelet, volume_wavelet)
            
            # Calculate Flow Pattern Recognition (FPR)
            def calculate_flow_pattern(p, v, window=20):
                pattern = pd.Series(index=p.index, dtype=float)
                for i in range(window, len(p)):
                    p_segment = p.iloc[i-window:i]
                    v_segment = v.iloc[i-window:i]
                    
                    # Calculate cross-spectral density
                    p_fft = np.fft.fft(p_segment)
                    v_fft = np.fft.fft(v_segment)
                    cross_spectrum = p_fft * np.conj(v_fft)
                    
                    # Extract magnitude and phase
                    magnitude = np.abs(cross_spectrum)
                    phase = np.angle(cross_spectrum)
                    
                    # Combine into pattern strength
                    pattern.iloc[i] = np.mean(magnitude) * np.cos(np.mean(phase))
                
                return pattern
            
            fpr = calculate_flow_pattern(price_changes, volume_changes)
            
            # Normalize components
            lws_norm = (lws - lws.rolling(window=20).mean()) / lws.rolling(window=20).std()
            hri_norm = (hri - hri.rolling(window=20).mean()) / hri.rolling(window=20).std()
            fpr_norm = (fpr - fpr.rolling(window=20).mean()) / fpr.rolling(window=20).std()
            
            # Calculate oscillation strength
            osc_strength = price_changes.rolling(window=20).std() / \
                          price_changes.rolling(window=60).std()
            
            # Dynamic weights based on oscillation strength
            w1 = 0.4 * (1 - osc_strength)  # More weight on LWS in low oscillation
            w2 = 0.3 * (1 + osc_strength)  # More weight on HRI in high oscillation
            w3 = 1 - w1 - w2               # Remaining weight to FPR
            
            # Combine signals with non-linear transformation
            result = np.tanh(w1 * lws_norm + 
                           w2 * hri_norm + 
                           w3 * fpr_norm)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result

        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])