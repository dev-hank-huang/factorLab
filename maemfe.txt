# distutils: language = c++
cimport numpy as np
import numpy as org_np
from libc.math cimport isnan
from libcpp.vector cimport vector
from libcpp.map cimport map as map
from cython.operator import dereference, postincrement
# import matplotlib.pyplot as plt

cdef np.ndarray price = org_np.zeros((0,0),  dtype=float)
cdef map[int, int] pos2price

trades = []
trade_positions = []
mae_mfe = []

cdef np.ndarray entry = org_np.zeros(0, dtype=int)
cdef np.ndarray position = org_np.zeros(0, dtype=float)

cdef int window
cdef int window_step
cdef int date

cdef void record_date(int d):
  global date
  date = d

cdef void start_analysis(
  np.ndarray[np.float64_t, ndim=2] price_, 
  map[int, int] pos2price_, 
  int nstocks, int window_, int window_step_):
  
  global price, pos2price, window, window_step
  price = price_
  pos2price = pos2price_
  window = window_
  window_step = window_step_
  
  global trades, trade_positions, mae_mfe
  trades.clear()
  trade_positions.clear()
  mae_mfe.clear()
  
  global entry, position
  entry = org_np.zeros(nstocks, dtype=int)
  position = org_np.zeros(nstocks, dtype=float)
  
cdef end_analysis(map[int, double] &pos):

  global trades, trade_positions, mae_mfe, date
  cdef int sid = 0

  cdef map[int, double].iterator it = pos.begin()
  while it != pos.end():

    # To avoid duplication of exit records, 
    # we need to handle the case where the exit has already been touched. 
    # If an exit has been touched, it means that its position in the 
    # system will be set to zero and the record_exit function will be executed, 
    # which will result in a duplicate exit record being created. 
    # Therefore, we need to handle this scenario to prevent duplication of exit records.
    if dereference(it).second != 0:
        sid = dereference(it).first

        # record exit for existing assets
        trades.append([sid, entry[sid], -1])
        trade_positions.append(position[sid])
        mae_mfe.append(mae_mfe_analysis(sid, entry[sid], date, position[sid] > 0))

    postincrement(it)


cdef void record_entry(int sid, double position_sid):
  global entry, position, date
  entry[sid] = date
  position[sid] = position_sid
  
cdef void record_exit(int sid):
    
  global trades, trade_positions, mae_mfe, entry, window, date
  
  trades.append([sid, entry[sid], date])
  trade_positions.append(position[sid])
  mae_mfe.append(mae_mfe_analysis(sid, entry[sid], date, position[sid] > 0))

cdef np.ndarray mae_mfe_analysis(int sid, int i_entry, int i_exit,  is_long):
  
#   print(sid, i_entry, i_exit, '-------------------')
    
  global price, pos2price, window, window_step, entry, date

  cdef int pid = pos2price[sid]
  
  cdef double price_ratio = 1
  
  cdef vector[double] cummax, cummin, mdd, profit_period, returns
  cdef vector[int] cummin_i
  
  i_exit_max = i_exit
  if window + i_entry > i_exit_max:
    i_exit_max = window + i_entry
    
#   print('i_exit_max 1', i_exit_max)
    
  cdef int plength = (<object> price).shape[0]
  
  if i_exit_max >= plength:
    i_exit_max = plength-1
    
  cummax.reserve(i_exit_max - i_entry + 1)
  cummin.reserve(i_exit_max - i_entry + 1)
  cummin_i.reserve(i_exit_max - i_entry + 1)
  mdd.reserve(i_exit_max - i_entry + 1)
  profit_period.reserve(i_exit_max - i_entry + 1)
  returns.reserve(i_exit_max - i_entry + 1)
  
  cummax.push_back(1)
  cummin.push_back(1)
  cummin_i.push_back(0)
  mdd.push_back(0)
  profit_period.push_back(0)
  returns.push_back(1)
  
  cdef double v = 1
  cdef double pv = price[i_entry][pid]
  cdef double p = price[i_entry][pid]
  cdef int i = 0
  #print('----', i_entry, pid, price[i_entry-1][pid], price[i_entry][pid], price[i_entry+1][pid])
  price_ratio = 1
  
  for i, ith in enumerate(range(i_entry+1, i_exit_max+1)):

    p = price[ith][pid]

    if not isnan(p):
      v = p / pv if is_long else pv / p
      pv = p
    
      if not isnan(v):
        price_ratio *= v

    cmax = cummax[i]
    cmin = cummin[i]
    if price_ratio > cmax:
      cummax.push_back(price_ratio)
      cmax = price_ratio
    else:
      cummax.push_back(cmax)
      
    if price_ratio < cmin:
      cummin.push_back(price_ratio)
      cummin_i.push_back(i)
    else:
      cummin.push_back(cmin)
      cummin_i.push_back(cummin_i[i])

    newmdd = price_ratio / cmax - 1
    if newmdd < mdd[i]:
        mdd.push_back(newmdd)
    else:
        mdd.push_back(mdd[i])

    profit_period.push_back(profit_period[i] + (price_ratio > 1))
    returns.push_back(price_ratio)
  
#   plt.plot(range((cummax.size())), cummax)
#   plt.plot(range((cummax.size())), cummin)
#   plt.plot(range((cummax.size())), value_price)
#   print(cummax)
#   print('---')
#   print(cummin)
#   print('---')
#   print(value_price)
#   plt.show()
  
  cdef int arsize = ((window-1) // window_step + 2) * 6
  cdef np.ndarray[np.float64_t, ndim=1] ret = org_np.empty(arsize)
  ret.fill(-1)
  
  # maes = []
  # gmfes = []
  # bmfes = []
  
  i = 0    
  for w in range(0, min(cummax.size(), window), window_step):
    mae = cummin[w] - 1
    gmfe = cummax[w] - 1
    
    mae_i = cummin_i[w]
    bmfe = cummax[mae_i] - 1
    ret[i] = mae
    i+=1
    ret[i] = gmfe
    i+=1
    ret[i] = bmfe
    i+=1
    ret[i] = mdd[w]
    i+=1
    ret[i] = profit_period[w]
    i+=1
    ret[i] = returns[w] - 1
    i+=1
    # maes.append(mae)
    # gmfes.append(gmfe)
    # bmfes.append(bmfe)

  # plt.plot(range(len(maes)), maes)
  # plt.plot(range(len(maes)), gmfes)
  # plt.plot(range(len(maes)), bmfes)
  # plt.show()
    
  w = min(i_exit - i_entry, cummax.size()-1)
  mae = cummin[w] - 1
  gmfe = cummax[w] - 1
  mae_i = cummin_i[w]

  bmfe = cummax[mae_i] - 1

  ret_length = len(ret)

  ret[ret_length-6] = mae
  ret[ret_length-5] = gmfe
  ret[ret_length-4] = bmfe
  ret[ret_length-3] = mdd[w]
  ret[ret_length-2] = profit_period[w]
  ret[ret_length-1] = returns[w] - 1
      
  return ret

