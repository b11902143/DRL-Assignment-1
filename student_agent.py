import numpy as np
import pickle
import random
import gym

# ---------------------------
# 參數設定 (可自行調整)
# ---------------------------
ALPHA = 0.1    # 學習率
GAMMA = 0.99   # 折扣因子
EPSILON = 0.1  # epsilon-greedy 中隨機探索的機率

# ---------------------------
# 載入 Q-Table
# ---------------------------
# 在此嘗試從 "q_table.pkl" 讀取已經訓練好的 Q-Table
# 若讀取失敗，則將 Q-Table 初始化為空的 dict()
try:
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
except:
    Q_table = {}

# ---------------------------
# 取得動作的函式 (作業要求)
# ---------------------------
def get_action(obs):
    """
    這個函式會根據觀測狀態 obs，回傳一個動作(0~5)。
    其中若該 obs 沒有在 Q_table 中，就先用隨機方式挑選動作；
    若有在 Q_table，則用 ε-greedy 的方式挑選動作：
      - 以 EPSILON 的機率，使用隨機探索
      - 以 (1 - EPSILON) 的機率，從 Q-Table 取最高 Q 值的動作
    """
    # 如果該狀態還未出現在 Q-Table 中，先用一組全 0 或隨機初始化
    if obs not in Q_table:
        # 共有 6 種動作 [0,1,2,3,4,5] -> 初始化各動作的 Q-value 為 0.0
        Q_table[obs] = np.zeros(6)
        # 直接隨機選一個動作回傳（fallback策略）
        return random.choice([0, 1, 2, 3, 4, 5])

    # 若該 obs 已存在於 Q-Table，則採用 epsilon-greedy 決策
    if random.uniform(0, 1) < EPSILON:
        # 探索：隨機選擇動作
        return random.choice([0, 1, 2, 3, 4, 5])
    else:
        # 利用：選擇 Q-value 最大的動作
        state_action_values = Q_table[obs]  # shape: (6,)
        return int(np.argmax(state_action_values))
