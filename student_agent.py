import numpy as np
import pickle
import random
import gym
from simple_custom_taxi_env import SimpleTaxiEnv

# Global policy table: state (tuple) -> logits (np.array with shape (6,))
policy_table = {}

def softmax(x):
    """計算 softmax（數值穩定版本）"""
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def transform_state(obs, env=None):
    """
    將原始觀察值 obs 轉換成新的 state 表示，其內容包含：
      - 四個 station 相對於 taxi 的位置 (station_row - taxi_row, station_col - taxi_col)
        (共 8 個數字)
      - 當 taxi 位於某個 station 上時，根據 obs[14] 與 obs[15] 設置：
            若 obs[14] 為 1，表示可執行 pickup；否則 0
            若 obs[15] 為 1，表示可執行 dropoff；否則 0
        (共 2 個 flag)
      - 障礙物資訊：obs[10:14] (共 4 個數值)
    最終 state 為長度 14 的 tuple
    """
    taxi_row, taxi_col = obs[0], obs[1]
    rel_positions = []
    for i in range(4):
        station_row = obs[2 + 2*i]
        station_col = obs[2 + 2*i + 1]
        rel_positions.extend([station_row - taxi_row, station_col - taxi_col])
    
    at_station = any(taxi_row == obs[2 + 2*i] and taxi_col == obs[2 + 2*i + 1] for i in range(4))
    pickup_possible = obs[14] if at_station else 0
    dropoff_possible = obs[15] if at_station else 0

    obstacles = list(obs[10:14])
    
    new_state = tuple(rel_positions + [pickup_possible, dropoff_possible] + obstacles)
    return new_state

def transform_state_with_passenger(obs, env=None):
    # 此表示已包含 pickup/dropoff flag
    return transform_state(obs, env)

def get_action(obs, epsilon=0.1):
    """
    根據觀察值 obs 轉換 state，利用 ε-greedy 策略選擇動作：
      - 以 epsilon 機率選隨機動作，
      - 否則根據 policy_table 中該 state 的 logits 使用 softmax 得到的機率選動作。
      - 若 state 尚未出現，則直接隨機選擇。
    """
    state = transform_state_with_passenger(obs, None)
    if np.random.rand() < epsilon or state not in policy_table:
        return random.choice([0, 1, 2, 3, 4, 5])
    else:
        logits = policy_table[state]
        probs = softmax(logits)
        action = np.random.choice(6, p=probs)
        return int(action)

def explore_and_record_stations_episode(env, init_obs, bonus_reward=50, epsilon=0.1):
    """
    探索階段：
      - 利用 get_action (含 ε-greedy) 選擇動作，讓 taxi 嘗試朝向各個 station 移動。
      - 當 taxi 與某個 station 的相對位置為 (0,0) 時，
        根據 obs[14] 與 obs[15] 記錄該 station 為 passenger 或 destination，
        並額外給予 bonus_reward 的獎勵（僅第一次拜訪）。
      - 採用 reward shaping：以基底 shaping reward 為 -0.06，再在達到 station 時增加 bonus_reward。
      - 以 env.step() 返回的 done 判斷探索是否結束。
    傳回包含 station 絕對位置、passenger/destination station、最終 obs 與探索軌跡的 dict。
    """
    discovered_indices = set()
    passenger_station = None
    destination_station = None
    obs = init_obs
    done = False
    step_count = 0
    exploration_trajectory = []  # 記錄 (state, action, reward) 的列表

    while len(discovered_indices) < 4 and not done:
        state = transform_state_with_passenger(obs, env)
        action = get_action(obs, epsilon)
        new_obs, raw_reward, done, _ = env.step(action)
        
        # Reward shaping：先以 -0.06 為基底
        shaped_reward = -0.06
        
        new_state = transform_state(new_obs, env)
        taxi_row, taxi_col = new_obs[0], new_obs[1]
        for i in range(4):
            rel_r = new_state[2*i]
            rel_c = new_state[2*i+1]
            # 當 taxi 與 station 重合且該 station 尚未被探索過
            if rel_r == 0 and rel_c == 0 and i not in discovered_indices:
                discovered_indices.add(i)
                shaped_reward += bonus_reward
                if new_obs[14] == 1 and passenger_station is None:
                    passenger_station = (new_obs[2 + 2*i], new_obs[2 + 2*i + 1])
                if new_obs[15] == 1 and destination_station is None:
                    destination_station = (new_obs[2 + 2*i], new_obs[2 + 2*i + 1])
        
        final_reward = raw_reward + shaped_reward
        exploration_trajectory.append((state, action, final_reward))
        obs = new_obs
        step_count += 1

    stations = [(init_obs[2 + 2*i], init_obs[2 + 2*i + 1]) for i in range(4)]
    return {
        "stations": stations,
        "passenger": passenger_station,
        "destination": destination_station,
        "final_obs": obs,
        "exploration_trajectory": exploration_trajectory,
    }

def train_agent(num_episodes=10000, alpha=0.01, gamma=0.99,
                initial_epsilon=1.0, min_epsilon=0.1, decay_rate=0.9996):
    """
    訓練流程分兩階段：
      1. 探索階段：使用 ε-greedy 策略探索，記錄 passenger 與 destination station，
         並在 taxi 第一次拜訪 station 時給予額外 bonus reward，
         同時採用 reward shaping 方式（基底 -0.06）。
      2. 任務階段：依據探索資訊進行接送任務，同樣以 ε-greedy 選動作，
         並利用 REINFORCE 方式更新 policy_table，
         任務階段也採用 reward shaping 方式（基底 -0.06 加上各種調整）。
    訓練過程中，ε 逐步衰減，從較高探索率轉向較低探索率。
    同時累計統計探索階段成功找到 passenger 與 destination 的次數，
    並每 100 個 episode 印出當前的累計結果。
    """
    global policy_table
    env = SimpleTaxiEnv()
    rewards_per_episode = []
    epsilon = initial_epsilon

    # 統計找到 passenger 與 destination 的次數
    passenger_found_count = 0
    destination_found_count = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        # 探索階段，記錄探索軌跡
        station_info = explore_and_record_stations_episode(env, obs, bonus_reward=50, epsilon=epsilon)
        passenger_station = station_info["passenger"]
        destination_station = station_info["destination"]
        if passenger_station is not None:
            passenger_found_count += 1
        if destination_station is not None:
            destination_found_count += 1

        if passenger_station is None or destination_station is None:
            print(f"Episode {episode+1}: 未完整發現 passenger 或 destination station，跳過任務階段")
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}: Passenger found {passenger_found_count/(episode+1):.2f} times, Destination found {destination_found_count/(episode+1):.2f} times")
            continue

        # 取得探索階段的軌跡
        exploration_trajectory = station_info["exploration_trajectory"]

        # 任務階段
        obs = station_info["final_obs"]
        state = transform_state_with_passenger(obs, env)
        done = False
        task_trajectory = []  # 記錄 (state, action, reward)
        total_reward = 0
        
        while not done:
            if state not in policy_table:
                policy_table[state] = np.zeros(6)
            logits = policy_table[state]
            probs = softmax(logits)
            if np.random.rand() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                action = np.random.choice(6, p=probs)
            
            pre_taxi_pos = (obs[0], obs[1])
            pre_passenger = env.passenger_picked_up
            target_pos = passenger_station if not pre_passenger else destination_station
            old_distance = abs(obs[0] - target_pos[0]) + abs(obs[1] - target_pos[1])
            
            new_obs, raw_reward, done, _ = env.step(action)
            new_state = transform_state_with_passenger(new_obs, env)
            new_distance = abs(new_obs[0] - target_pos[0]) + abs(new_obs[1] - target_pos[1])
            
            # Reward shaping：以 -0.06 為基底
            shaped_reward = -0.06
            shaped_reward += (old_distance - new_distance) * 0.5
            
            if any(flag == 1 for flag in obs[10:14]):
                shaped_reward -= 5
            
            taxi_row, taxi_col = obs[0], obs[1]
            if taxi_row == 0 and action == 1:
                shaped_reward -= 2
            if taxi_row == env.grid_size - 1 and action == 0:
                shaped_reward -= 2
            if taxi_col == 0 and action == 3:
                shaped_reward -= 2
            if taxi_col == env.grid_size - 1 and action == 2:
                shaped_reward -= 2

            if action == 4:  # PICKUP
                if not pre_passenger and pre_taxi_pos == passenger_station:
                    shaped_reward += 2
                else:
                    shaped_reward -= 2
            if action == 5:  # DROPOFF
                if pre_passenger and pre_taxi_pos == destination_station:
                    shaped_reward += 3
                else:
                    shaped_reward -= 2
            
            final_reward = raw_reward + shaped_reward
            total_reward += final_reward
            task_trajectory.append((state, action, final_reward))
            state = new_state
            obs = new_obs
        
        rewards_per_episode.append(total_reward)
        
        # 合併探索階段與任務階段的軌跡
        full_trajectory = exploration_trajectory + task_trajectory

        # REINFORCE 更新 policy：從後往前累積回報並更新梯度
        G = 0
        for s, a, r in reversed(full_trajectory):
            G = r + gamma * G
            if s not in policy_table:
                policy_table[s] = np.zeros(6)
            logits = policy_table[s]
            probs = softmax(logits)
            one_hot = np.zeros(6)
            one_hot[a] = 1
            gradient = one_hot - probs
            policy_table[s] += alpha * G * gradient
        
        # 每個 episode 衰減 epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, total reward: {total_reward}, epsilon: {epsilon:.3f}")
            print(f"Episode {episode+1}: Passenger found {passenger_found_count/(episode+1):.2f} times, Destination found {destination_found_count/(episode+1):.2f} times")
            
    try:
        env.close()
    except Exception as e:
        pass
    return policy_table, rewards_per_episode

# 優先嘗試讀取已訓練的 policy；若不存在則進行訓練並存檔
try:
    with open("policy.pkl", "rb") as f:
        policy_table = pickle.load(f)
    print("Loaded trained policy from policy.pkl")
except Exception as e:
    print("Training new policy using policy gradient with epsilon and reward shaping...")
    policy_table, rewards = train_agent(num_episodes=10000, alpha=0.01, gamma=0.99,
                                         initial_epsilon=1.0, min_epsilon=0.1, decay_rate=0.9996)
    with open("policy.pkl", "wb") as f:
        pickle.dump(policy_table, f)
    print("Training complete and policy saved to policy.pkl")
