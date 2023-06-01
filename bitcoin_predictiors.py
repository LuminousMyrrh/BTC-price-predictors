import requests
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load

import time
import os
import threading
import json

closed_prices = []
act_prices = []
btc_volume = []


def actual_prices(interval, symbol="BTCUSDT"):
    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 50

# Делаем запрос к API Binance и обрабатываем ответ
    response = requests.get(url, params={"symbol": symbol, "interval": f'{interval}', "limit": limit})
    if response.status_code == 200:
        klines = response.json()
        for kline in klines:
            # Получаем цену закрытия из свечи
            close_price = float(kline[4])
            # Добавляем цену в массив
            act_prices.append(close_price)

def get_price(symbol="BTCUSDT"):
    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
    while True:
        response = requests.get(url)
        data = response.json()
        price = float(data['price'])
        act_prices.append(price)
        if len(act_prices) >= 51:
            act_prices.pop(0)

def update_closed_prices(interval, symbol="BTCUSDT"):
    while True:
        # Запрос на получение данных по закрытым свечам биткоина
        response = requests.get('https://fapi.binance.com/fapi/v1/klines', params={
            'symbol': f'{symbol}',
            'interval': f'{interval}',
            'limit': 1
        })

    # Парсим полученный JSON-ответ и добавляем закрытую цену в массив
        if response.status_code == 200:
            kline_data = json.loads(response.text)[0]
            closed_price = float(kline_data[4])
            volume = float(kline_data[5])
            btc_volume.append(volume)
            closed_prices.append(closed_price)
            if len(closed_prices) >= 51:
                    closed_prices.pop(0)
            if len(btc_volume) >= 151:
                btc_volume.pop(0)

def get_closed_prices(interval, symbol="BTCUSDT"):
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': f'{symbol}', 'interval': f'{interval}', 'limit': 150}

    # Выполняем GET-запрос и получаем ответ в формате JSON
    response = requests.get(url, params=params)
    data = response.json()

    for candle in data:
        closing_price = float(candle[4])
        closed_prices.append(closing_price)
        vol = float(candle[5])
        btc_volume.append(vol)

def SMA(period, prices):
    return sum(prices[-period:]) / period

def MA(period, closed_prices):
    sum_prices = sum(closed_prices[-period:])
    ma = sum_prices / period
    return ma

def EMA(period, prices):
    k = 2 / (period + 1)
    ema = SMA(period, prices)
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * k + ema
    return ema


def SD(closed_prices, period=50, calculation_type='simple'):
    sd = 0
    # Ограничиваем количество последних цен заданным периодом
    closed_prices = closed_prices[-period:]

    # Вычисляем SD в зависимости от типа расчета
    if calculation_type == 'simple':
        sd = np.std(closed_prices)
    elif calculation_type == 'sma':
        sma = sum(closed_prices) / period
        deviations = [(price - sma) ** 2 for price in closed_prices]
        sd = np.sqrt(sum(deviations) / period)
    elif calculation_type == 'ema':
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        sma = np.convolve(closed_prices, weights, mode='valid')[0]
        deviations = [(price - sma) ** 2 for price in closed_prices[-period+1:]]
        sd = np.sqrt(sum(deviations) / period)

    return sd

def MACD(closed_prices):
    # Вычисляем две скользящие средние
    fast_ema = EMA(8, closed_prices)
    slow_ema = EMA(21, closed_prices)
    # Вычисляем разницу между ними
    macd = fast_ema - slow_ema
    # Вычисляем EMA от разницы
    signal_line = EMA(5, [macd])
    # Вычисляем разницу между MACD и сигнальной линией
    histogram = macd - signal_line
    return macd, signal_line, histogram

def RSI(period=14):
    delta_prices = np.diff(closed_prices)
    gains = delta_prices[delta_prices >= 0]
    losses = -delta_prices[delta_prices < 0]
    avg_gain = SMA(period, gains)
    avg_loss = SMA(period, losses)

    # Calculate RSI
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # Clip RSI to between 0 and 100
    rsi = np.clip(rsi, 0, 100)
    
    return rsi

def CRSI(period=3, rsi_period=2, ma_type='ema', calculation_type='simple'):
    global act_prices
    # Вычисляем RSI
    rsi = RSI(rsi_period)
    # Вычисляем критерии
    sma = SMA(period, closed_prices)
    std_dev = SD(closed_prices, period=period, calculation_type=calculation_type)
    # Вычисляем верхнюю и нижнюю границы
    upper_band = sma + (std_dev * 0.5)
    lower_band = sma - (std_dev * 0.5)
    # Вычисляем критерий 1
    criteria_1 = ((rsi - 50) / 10) + 0.5
    # Вычисляем критерий 2
    if act_prices[-1] > upper_band:
        criteria_2 = -1
    elif act_prices[-1] < lower_band:
        criteria_2 = 1
    else:
        criteria_2 = 0
    # Вычисляем критерий 3
    if ma_type == 'ema':
        ma = EMA(period, closed_prices)
    else:
        ma = MA(period, closed_prices)
    if act_prices[-1] > ma:
        criteria_3 = -1
    else:
        criteria_3 = 1
    # Вычисляем Connors RSI
    connors_rsi = (criteria_1 + criteria_2 + criteria_3) / 3 * 100
    return connors_rsi

def BollingerBands(closed_prices, period=20, std_dev=2):
    middle_band = MA(period, closed_prices)
    sd = SD(closed_prices, period)
    upper_band = middle_band + (std_dev * sd)
    lower_band = middle_band - (std_dev * sd)

    return upper_band, middle_band, lower_band

def OBV(period=9, prices=closed_prices, volumes=btc_volume):
    obv = 0
    prices = prices[-period:]
    volumes = volumes[-period:]
    
    # Вычисляем изменение цен и объемов за каждую свечу
    changes = [1 if prices[i] > prices[i-1] else -1 if prices[i] < prices[i-1] else 0 for i in range(1, len(prices))]
    volumes = volumes[1:]
    
    # Считаем сумму объемов с учетом изменения цены
    for i in range(len(changes)):
        obv += volumes[i] * changes[i]
    return obv

def STOCH(closed_prices, period=14, sma_period=3):
    # Ограничиваем количество последних цен заданным периодом
    closed_prices = closed_prices[-period:]
    # Вычисляем минимальную и максимальную цены за период
    min_price = min(closed_prices)
    max_price = max(closed_prices)
    # Вычисляем %K
    k = ((closed_prices[-1] - min_price) / (max_price - min_price)) * 100
    # Вычисляем среднюю скользящую %K за период
    k_sma = SMA(sma_period, [k])
    # Вычисляем среднюю скользящую %D за период
    d = SMA(sma_period, [k_sma])
    return k_sma, d

def RVI(closed_prices, period=14):
    rvi = 0
    avg_up = 0
    avg_down = 0

    for i in range(1, len(closed_prices)):
        price_change = closed_prices[i] - closed_prices[i-1]
        if price_change > 0:
            avg_up = (avg_up * (period - 1) + price_change) / period
            avg_down = (avg_down * (period - 1)) / period
        else:
            avg_up = (avg_up * (period - 1)) / period
            avg_down = (avg_down * (period - 1) - price_change) / period

    if avg_down == 0:
        rvi = 100
    else:
        rvi = 100 * avg_up / (avg_up + avg_down)

    if rvi > 80:
        return "High volatility"
    elif rvi < 20:
        return "Low volatility"
    elif rvi >= 20 and rvi <= 80:
        return "Average volatility"
    else:
        return "No volatility"

try:
    model_forest = load('model_forest.joblib')
    print("model forest loaded")
except:
    model_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    print("model forest created")
try:
    model_grad = load('model_grad.joblib')
    print("model glad loaded")
except:
    model_grad = GradientBoostingRegressor(n_estimators=100, random_state=42)
    print("model glad created")
try:
    model_multi = load('model_multi.joblib')
    print("model multi loaded")
except:
    model_multi = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5))
    print("model multi created")

# Define the stacking regressor
stacking = StackingRegressor(
    estimators=[('forest', model_forest), ('grad', model_grad)],
    final_estimator=model_forest
)
time.sleep(1)

def prepare_data(closed_prices, sma, ma, ema, sd, macd, sig, hist, rsi, crsi, up, mid, low, obv, ks, ds):
    X = []
    y = []
    X.append(closed_prices)
    X.append(sma)
    X.append(ma)
    X.append(ema)
    X.append(sd)
    X.append(macd)
    X.append(sig)
    X.append(hist)
    X.append(rsi)
    X.append(crsi)
    X.append(up)
    X.append(mid)
    X.append(low)
    X.append(obv)
    X.append(ks)
    X.append(ds)
    # Задаем целевую переменную
    #y = act_prices[-1]
    for i in range(1, 16):
        y.append(float(closed_prices[-i]))
    y.append(float(act_prices[-1]))

    # Делим данные на обучающую и тестовую выборки в соотношении 75:25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=42)
    features = act_prices[-1]
    features = np.array(features).reshape(1, -1)

    return X_train, y_train, X_test, y_test, features


def letgo(models):
    sma = SMA(7, closed_prices)
    ma = MA(9, closed_prices)
    ema = EMA(9, closed_prices)
    sd = SD(closed_prices)
    macd, sig, hist = MACD(closed_prices)
    rsi = RSI()
    crsi = CRSI()
    up, mid, low = BollingerBands(closed_prices)
    obv = OBV()
    ks, ds = STOCH(closed_prices)

    xtr, ytr, _, _, features = prepare_data(closed_prices, sma, ma, ema, sd, macd, sig, hist, rsi, crsi, up, mid, low, obv, ks, ds)

    xtr = np.array(xtr).reshape(-1, 1)

    models.fit(xtr, ytr)
    y_pred = models.predict(features)
    return round(y_pred[-1], 1)


def start():
    inter = '4h'
    symbol = 'BTCUSDT'

    os.system('clear')
    user_inter = input('Choose interval:\n1. 1m\n2. 5m\n3. 15m\n4. 30m\n5. 1h\n6. 2h\n7. 4h\n8. 12h\n9. 1D\n10. 1W\n11. 1M\n\nInterval: ')
    if user_inter == '1':
        inter = '1m' 
    if user_inter == '2':
        inter = '5m' 
    if user_inter == '3':
        inter = '15m' 
    if user_inter == '4':
        inter = '30m' 
    if user_inter == '5':
        inter = '1h' 
    if user_inter == '6':
        inter = '2h'
    if user_inter == '7':
        inter = '4h'
    if user_inter == '8':
        inter = '12h'
    if user_inter == '9':
        inter = '1d'
    if user_inter == '10':
        inter = '1w'
    if user_inter == '11':
        inter = '1M'

    os.system('clear')
    print(f"Your interval: {inter}\n")
    user_symbol = input('Choose pair:\n1. BTCBUSD\n2. BTCUSDT\n3. ETHUSDT\n4. ETHBUSD\n5. DODOBUSD\n6. APTUSDT\n7. PEOPLEUSDT\n\nSymbol: ')

    if user_symbol == '1':
        symbol = 'BTCUSDT'
    if user_symbol == '2':
        symbol = 'BTCUSDT'
    if user_symbol == '3':
        symbol = 'ETHUSDT'
    if user_symbol == '4':
        symbol = 'ETHBUSD'
    if user_symbol == '5':
        symbol = 'DODOBUSD'
    if user_symbol == '6':
        symbol = 'APTUSDT'
    if user_symbol == '7':
        symbol = 'PEOPLEUSDT'
    if user_symbol == '8':
        symbol = 'ARPAUSDT'

    get_closed_prices(inter, symbol)

    def run_all():
        get_price(symbol)
        update_closed_prices(inter, symbol)

    try:
        threading.Thread(target=run_all).start()
    except Exception as e:
        print(e)

    time.sleep(2)

    while True:
        try:
            last_closed = closed_prices
            pred1 = letgo(model_forest)
            pred2 = letgo(model_grad)
            pred3 = letgo(stacking)
            diff = 0

            dump(model_forest, 'model_forest.joblib')
            dump(model_grad, 'model_grad.joblib')
            dump(model_multi, 'model_multi.joblib')
            middle = (pred1 + pred2 + pred3) // 3

            price = act_prices[-1]
            diff = price - middle
            if diff < 0:
                diff = middle - price
            
            if middle > price:
                where = 'UP✅'
            else:
                where =  'DOWN🔻'

            os.system('clear')
            print("Prediction from RandomForest: ", pred1)
            print("Prediction from GradientBoosting: ", pred2)
            print("Prediction from stacked model: ", pred3)
            print(f"\nInterval: {inter}")
            print(f"Pair: {symbol}")
            print(f"\nMid price: {middle}")
            print(f"Difference between prices: {round(diff)}")
            print("Actual price: ", act_prices[-1])
            print("Last closed price: ", closed_prices[-1])
            print(f"Price will be go {where}")
            if closed_prices[-1] == last_closed:
                time.sleep(55)
            else:
                continue
        except Exception as e:
            print(e)


start()
'''
try:
    threading.Thread(target=start).start()
except Exception as e:
    print(e)
'''
