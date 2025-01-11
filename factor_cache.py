from typing import Any
from datetime import datetime
import sys
from multiprocessing import shared_memory
import numpy as np

class MemoryCache:
    def __init__(self, max_cache_size_gb=48):
        self._cache = {}  # 使用本地字典
        self._shared_blocks = {}  # 儲存共享記憶體區塊的資訊
        self._max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024
        self._current_cache_size = 0

    def get(self, key: str, max_age_minutes: int = 60) -> Any:
        if key not in self._cache:
            return None
            
        try:
            # 通過共享記憶體取得資料
            shm = shared_memory.SharedMemory(name=self._shared_blocks[key]['name'])
            data = np.ndarray(
                self._shared_blocks[key]['shape'],
                dtype=self._shared_blocks[key]['dtype'],
                buffer=shm.buf
            )
            return data
        except Exception as e:
            print(f"快取讀取錯誤: {str(e)}")
            return None
        finally:
            if 'shm' in locals():
                shm.close()

    def set(self, key: str, value: Any) -> None:
        try:
            if isinstance(value, np.ndarray):
                # 創建共享記憶體區塊
                shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
                shared_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
                shared_array[:] = value[:]
                
                # 儲存共享記憶體區塊的資訊
                self._shared_blocks[key] = {
                    'name': shm.name,
                    'shape': value.shape,
                    'dtype': value.dtype
                }
                
                self._cache[key] = True  # 標記該鍵值已存在
                
        except Exception as e:
            print(f"快取設置錯誤: {str(e)}")

    def clear(self):
        for key in self._shared_blocks:
            try:
                shm = shared_memory.SharedMemory(name=self._shared_blocks[key]['name'])
                shm.close()
                shm.unlink()
            except Exception:
                continue
        self._cache.clear()
        self._shared_blocks.clear()