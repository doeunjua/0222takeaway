# 0222takeaway
## **takeaway1. 신경망의 입력과 출력 생각해보기**
![image](https://github.com/doeunjua/0222takeaway/assets/122878319/f80f63b6-e74c-46de-86cd-fc32327d7fe8)

오늘 배웠던 그림인데 입력으로 뭐가 들어갈지 출력으로 뭐가 나올지 감이 잡히나요? 한번 고민해보세요

<img width="770" alt="image" src="https://github.com/doeunjua/0222takeaway/assets/122878319/acec9c23-8d40-4a96-a9f7-926a04973596">

오늘 배운 그림에서 입력으로 뭐가 들어가고 출력으로 뭐가 나오는지 표시한 그림입니다. 
오늘 배운 sequential과 fit활용한 코드와 화요일에 재필오빠 시간에 배운 call써서 하는 코드로 위의 신경망을 어떻게 구성했는지 확인해보세요

[코드1]
```python
def _build_model(self):
        
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size,   ))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
```
[코드2]
```python
super(DQN, self).__init__()
        self.d1 = Dense(16, input_dim=16, activation='relu')
        self.d2 = Dense(16, activation='relu')
        self.d3 = Dense(4, activation='linear')
        self.optimizer = Adam(0.001)

        self.M = []  # M은 리플레이 버퍼

    def call(self, x): # x는 넘파이 어레이
        x = self.d1(x)
        x = self.d2(x)
        y_hat = self.d3(x)
        return y_hat
```
## **takeaway2. DQN알고리즘 이해하기**
DQN을 시작하고나서 알고리즘을 전체적으로 본 적이 없는 것 같아요

<img width="572" alt="image" src="https://github.com/doeunjua/0222takeaway/assets/122878319/b34ea8c3-93a2-48b1-bbd9-b6dce57255f7">

이 알고리즘은 DQN 방식을 처음으로 발표한 논문에 나오는 알고리즘이에요
아래는 이 알고리즘에 대한 gpt의 설명입니다.

<img width="503" alt="image" src="https://github.com/doeunjua/0222takeaway/assets/122878319/d48b17e5-e913-4974-9b3c-7139608c309d">

한번 연관지어 보시고 오늘 배운 코드랑도 연관지어 보세요 
class안의 함수랑 알고리즘이랑 연관지어서 어떻게 사용되는지 완벽하게는 아니더라도 이해해보세요

```python
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt



custom_map = [
    'FFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)#최대 2000개의 데이터를 저장할 수 있는 deque를 생성
        self.gamma = 0.95    
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model() #

    def _build_model(self):
        
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size,   ))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        #random.randrange(self.action_size) 또는 randint써도됨
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def one_hot_state(state):
    one_hot = np.zeros(state_size)
    one_hot[state] = 1
    return np.reshape(one_hot, [1, state_size])

agent = DQNAgent(state_size, action_size)


import matplotlib.pyplot as plt

EPISODES = 1000
BATCH_SIZE = 32  


rewards_per_episode = []

for e in range(EPISODES):
    state = env.reset()[0]
    state = one_hot_state(state)
    done = False
    total_reward = 0
    
    while not done:
       
        action = agent.act(state)
        next_state, reward, done, _ ,_= env.step(action)
        next_state = one_hot_state(next_state)
        # 게임이 진행중이거나 구멍에 빠지지 않았을 때를 위에 env.step(action)의 결과를 이용해서 어떻게 표현할까요?
        # 보상이 마지막에만 부여되어서 학습이 잘 안되므로 한 스텝 더 나아갈때마다중간중간 작은보상을 더 넣어준다
        if not done or reward == 0:  # 게임이 진행 중이거나 구멍에 빠지지 않았을 때
            reward += 0.01  # 작은 긍정적인 보상 부여 
            #이랬을 경우 어떤 문제가 생길까?reward-=0.01로 해준다면 어떻게 될까? 다시 미로로가서 음 주면 q
        if done and reward == 0:  # 구멍에 빠졌을 때
            reward -= 0.01
        
   
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    
    rewards_per_episode.append(total_reward)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode Over Time')
plt.show()
```

## **takeaway3. 보상수정**
오늘 dqn으로 학습시켰을 때 보상이 상태 15일때 1이 주어지는것으로는 학습이 잘 안된다고 제가 이야기 했었습니다. 그래서 보상을 수정했었는데 몇가지 다시 생각해볼거리를 적어봤어요

```python
if not done or reward == 0:  # 게임이 진행 중이거나 구멍에 빠지지 않았을 때
            reward += 0.01  # 작은 긍정적인 보상 부여 
        if done and reward == 0:  # 구멍에 빠졌을 때
            reward -= 0.01
```

처음에 이렇게 보상을 게임진행중이거나 구멍에 빠지지 않을때 양의 보상 그다음 구멍에 빠졌을 때 음의 보상 이렇게 주는 방식을 사용했었죠. 실제로 코드 돌려봤을때도 학습이 잘 되었습니다. 하지만 혹시나 더 큰 긍정보상을 받으려고 계속 빙글빙글돌다가 마지막 상태 15에 가는것을 목표로 할수도 있겠죠? 이 문제를 해결하기위해

```python
if not done or reward == 0:  # 게임이 진행 중이거나 구멍에 빠지지 않았을 때
            reward -= 0.01  # 작은 긍정적인 보상 부여 
            #이랬을 경우 어떤 문제가 생길까?reward-=0.01로 해준다면 어떻게 될까?
        if done and reward == 0:  # 구멍에 빠졌을 때
            reward -= 2
```
이런식으로 구멍에 빠지지 않도록 구멍에 빠질때 더 큰 음의 보상을 주고 게임을 잘 진행하고 있을때 -0.01의 보상을 부여해서 빙글빙글 돌지 않고 최단거리로 갈 수 있도록 설정해줍니다.

또 제가 fit함수 쓰지 않고 수동적으로 경사하강법 입력해서 빨리 돌아가게 만든 코드도 드렸는데 거기서는 입실론 그리디말고 소프트맥스 방법을 썼었던것 기억하시나요 거기서는 보상을 더 크게줬어요 그 이유 다시 떠떠올려보세요

```python
if done and reward == 0:
            reward = -5
        if not done or reward == 0:
            reward += 0.001
        if reward == 1:
            reward = 5
```

<img width="448" alt="image" src="https://github.com/doeunjua/0222takeaway/assets/122878319/ee3af9fb-2f4f-4929-968f-ddeea434e3e5">

위 그림처럼 선택할 수 있는 액션이 2개고 그것의 가치가 각각 1 5일때 두번째 행동의 가치가 첫번째 행동의 가치의 5배입니다. 소프트맥스 방법을 사용한다면 첫번째 행동 선택할 확률= 0.017986209962091555 그리고 두번째 행동 선택할 확률=0.9820137900379083 이렇게 되겠죠? 그러면 두 확률 나눠보면 거의 50배 정도 나오니까 두번째 행동 50번 선택할때 첫번째 행동 한번정도 선택한다는 의미에요 

근데 보상을 낮췄을때 가치가 첫번째 행동이 1이고 두번째 행동이 0.01라면 거의 100배차이나는 거잖아요 가치가! 아까는 5배 밖에 차이 안났는데... 근데 아래그림에서 볼 수 있듯이 소프트맥스쓰면 2.7배정도밖에 차이가 안나요. 그래서 거의 3번중에 1번은 가치가 낮은 행동이 선택된다는 거에요. 그러면 학습도 좀 잘 안될수 있겠죠. 그래서 입실론 탐욕방법 썼을때랑 다르게 보상 크게 설정해준거에요
<img width="453" alt="image" src="https://github.com/doeunjua/0222takeaway/assets/122878319/34351a64-65da-46e7-9140-fb18934b1f49">

그리고 제가 언급했었던 목표와 거리에 따라서 보상을 지정해주는 방식도 스스로 한번 짜보세요. 코드는 올려드릴게요

```python
def calculate_reward_adjustment(state, next_state):
    # 목표 위치 설정 (예: 4x4 환경에서 오른쪽 하단)
    goal_position = (3, 3)  # FrozenLake의 4x4 격자에서의 목표 위치
    
    # 현재 상태와 다음 상태의 좌표를 계산
    state_position = (state // 4, state % 4)
    next_state_position = (next_state // 4, next_state % 4)
    
    
    distance_to_goal_current = abs(goal_position[0] - state_position[0]) + abs(goal_position[1] - state_position[1])
    distance_to_goal_next = abs(goal_position[0] - next_state_position[0]) + abs(goal_position[1] - next_state_position[1])
    
    # 거리가 줄어들었으면 보상 증가
    if distance_to_goal_next < distance_to_goal_current:
        return 0.1  # 거리가 줄어들면 추가로 주는 보상
    else:
        return 0.0  # 거리가 줄어들지 않았다면 추가 보상 없음

# 에이전트의 행동을 취하는 부분에 보상 조정 로직 추가
for e in range(EPISODES):
    state = env.reset()
    state = one_hot_state(state)
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = one_hot_state(next_state)
        
        # 보상 조정
        reward_adjustment = calculate_reward_adjustment(state, next_state)
        reward += reward_adjustment
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
```

이 외에도 이 문제를 해결하기위해 더 추가해줄 수 있는 보상체계 아이디어가 떠오르신다면 직접 구현해보셔서 돌려보세요 학습이 잘 되는지!!
그리고 제가 빨리 돌아가는것 말고 처음으로 올려드린 코드에 내장함수 사용법이랑 여러 부가설명 자세하게 적어놨으니까 읽어보시면서 코드 이해해보세요.

오늘 너무 수고하셨어요!! dqn너무 어려워서 원래 한번에 이해 잘 안돼요 저는 1년 걸렸어요ㅎㅎ 대단하신거에요 굿굿
파이팅!!!!
