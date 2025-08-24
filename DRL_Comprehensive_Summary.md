# Summary of Shusen Wang's slides

The slides on Deep Reinforcement Learning are available on [Github](https://github.com/wangshusen/DRL/).

### Table of Contents


1. [Mathematical Foundations](#1-mathematical-foundations)

2. [Reinforcement Learning Basics](#2-reinforcement-learning-basics)

3. [Value-Based Methods](#3-value-based-methods)

4. [Temporal Difference Learning](#4-temporal-difference-learning)

5. [Deep Q-Networks (DQN)](#5-deep-q-networks-dqn)

6. [Policy Gradient Methods](#6-policy-gradient-methods)

7. [AlphaGo and Monte Carlo Tree Search](#7-alphago-and-monte-carlo-tree-search)

8. [Continuous Control](#8-continuous-control)

9. [Multi-Agent Reinforcement Learning](#9-multi-agent-reinforcement-learning)


---


## 1. Mathematical Foundations


### Random Variables and Probability


**Definition**: A random variable is unknown; its values depend on outcomes of random events.


**Notation**:

- Uppercase letter $X$ for a random variable

- Lowercase letter $x$ for an observed value


**Probability Mass Function (PMF)** for discrete distributions:

$$\sum_{x \in \mathcal{X}} p(x) = 1$$


**Probability Density Function (PDF)** for continuous distributions:

$$\int_{\mathcal{X}} p(x) dx = 1$$


### Key Concepts

- **Domain**: Random variable $X$ is in the domain $\mathcal{X}$

- **PDF provides relative likelihood** that the value of the random variable would equal that sample


---


## 2. Reinforcement Learning Basics


### Agent-Environment Interaction


The RL paradigm consists of:

- **Agent**: The learner/decision maker (e.g., Mario in Super Mario Bros)

- **Environment**: Everything the agent interacts with

- **State** $s_t$: Current situation of the agent

- **Action** $a_t$: Decision made by the agent

- **Reward** $r_t$: Feedback from the environment


### Core Terminology


#### Policy $\pi$

**Definition**: Policy function $\pi: (s, a) \mapsto [0,1]$


$$\pi(a  \vert  s) = \mathbb{P}(A = a  \vert  S = s)$$


The policy represents the probability of taking action $a$ in state $s$.


#### State Transition

- **State transition can be random**

- **Randomness comes from the environment**

- An action leads from old state to new state


#### Return (Cumulative Future Reward)

**Definition**: Return (aka cumulative future reward)


$$U_t = R_t + R_{t+1} + R_{t+2} + R_{t+3} + \cdots$$


**Question**: At time $t$, are $R_t$ and $R_{t+1}$ equally important?


**Discounted Return**:

$$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{n-t} R_n$$


where $\gamma$ is the discount factor ($0 < \gamma \leq 1$).


#### Action-Value Function $Q_\pi(s, a)$


**Definition**: Action-value function (Q-function)


$$Q_\pi(s_t, a_t) = \mathbb{E}[U_t  \vert  S_t = s_t, A_t = a_t]$$


This represents the expected discounted return when:

- Starting in state $s_t$

- Taking action $a_t$

- Following policy $\pi$ thereafter


### Practical Environment: OpenAI Gym


OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.

- Website: https://gym.openai.com/

- Provides classical control problems like:

- **Cart Pole**: Balance a pole on a cart

- **Pendulum**: Control an inverted pendulum


---


## 3. Value-Based Methods


### Deep Q-Network (DQN) Architecture


**Input**: Screenshots/observations (state $s$)

**Output**: Q-values for all possible actions


Architecture:

$$\text{State } s \rightarrow \text{Conv Layer} \rightarrow \text{Dense Layer} \rightarrow \begin{cases}
Q(s, \text{"left"}; w) \\
Q(s, \text{"right"}; w) \\
Q(s, \text{"up"}; w)
\end{cases}$$


**Key Properties**:

- Input shape: Size of the screenshot

- Output shape: Dimension of action space

- Uses convolutional layers for visual input processing

- Dense layers for final Q-value estimation


### Learning Process Example


**Scenario**: Driving from NYC to Atlanta

- Model $Q(\mathbf{w})$ estimates travel time (e.g., 1000 minutes)

- Actual travel time: 860 minutes

- **Question**: How do I update the model?


**Process**:

1. Make prediction: $q = Q(\mathbf{w}) = 1000$

2. Observe actual result: $y = 860$

3. Update model to reduce prediction error


---


## 4. Temporal Difference Learning


### TD Target and Learning


**Temporal Difference (TD) Learning** provides more reliable estimates:


**Example Continuation**:

- Model's estimate: $Q(\mathbf{w}) = 1000$ minutes

- Intermediate observation: NYC → DC takes 300 minutes (actual)

- Updated estimate: $300 + 600 = 900$ minutes (TD target)


**TD Target**: $y_t = r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})$


**Key Insight**: TD target $y = 900$ is more reliable than the original estimate of 1000.


**Loss Function**: $L = \frac{1}{2}(Q(\mathbf{w}) - y)^2$


**Gradient**: $\frac{\partial L}{\partial \mathbf{w}} = (1000 - 900) \cdot \frac{\partial Q(\mathbf{w})}{\partial \mathbf{w}}$


### SARSA Algorithm


**SARSA** (State-Action-Reward-State-Action) is an on-policy TD learning method.


**Algorithm Steps**:

1. Observe transition $(s_t, a_t, r_t, s_{t+1})$

2. Sample $a_{t+1} \sim \pi(\cdot  \vert  s_{t+1})$ using current policy

3. Compute TD target: $y_t = r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})$


**Tabular Version**:

For each state-action pair $(s, a)$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma Q(s', a') - Q(s, a)]$$


**Key Properties**:

- **On-policy**: Uses the same policy for action selection and evaluation

- Updates Q-values using the next action actually taken

- Converges to the optimal Q-function under certain conditions


### Derive TD Target


**Mathematical Derivation**:


Starting with the Bellman equation:

$$Q_\pi(s_t, a_t) = \mathbb{E}[U_t  \vert  S_t = s_t, A_t = a_t]$$


Using the recursive property:

$$Q_\pi(s_t, a_t) = \mathbb{E}[R_t + \gamma U_{t+1}  \vert  S_t = s_t, A_t = a_t]$$

$$= \mathbb{E}[R_t  \vert  S_t = s_t, A_t = a_t] + \gamma \mathbb{E}[Q_\pi(S_{t+1}, A_{t+1})  \vert  S_t = s_t, A_t = a_t]$$


This leads to the TD target:

$$y_t = r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})$$


---


## 5. Deep Q-Networks (DQN)


### Experience Replay


**Problem**: Waste of Experience

- A transition: $(s_t, a_t, r_t, s_{t+1})$

- **Experience**: All transitions over time

- Previously: Discard $(s_t, a_t, r_t, s_{t+1})$ after using it once

- **This is wasteful...**


**Solution**: Experience Replay

- Store transitions in a replay buffer

- Sample random batches for training

- Improves data efficiency and stability


**Benefits**:

1. **Data Efficiency**: Reuse experiences multiple times

2. **Stability**: Break correlation between consecutive samples

3. **Better Learning**: More diverse training batches


### DQN Improvements


**Key Innovations**:

1. **Experience Replay**: Store and reuse transitions

2. **Target Network**: Separate network for computing targets

3. **Clipping**: Clip gradients for stability


**DQN Algorithm**:

1. Collect experience $(s_t, a_t, r_t, s_{t+1})$

2. Store in replay buffer

3. Sample mini-batch from buffer

4. Compute TD targets using target network

5. Update main network via gradient descent


---


## 6. Policy Gradient Methods


### Baseline Methods


**Problem**: High variance in policy gradient estimates


**Solution**: Use a baseline $b$ that is independent of action $A$:


$$\mathbb{E}_{A \sim \pi}\left[b \cdot \frac{\partial \ln \pi(A  \vert  s;\theta)}{\partial \theta}\right] = b \cdot \mathbb{E}_{A \sim \pi}\left[\frac{\partial \ln \pi(A  \vert  s;\theta)}{\partial \theta}\right]$$


**Key Property**: The baseline doesn't introduce bias but reduces variance.


**Common Baselines**:

- Constant baseline

- State-value function $V(s)$

- Moving average of returns


### Policy Gradient Theorem


**Objective**: Maximize expected return

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$


**Policy Gradient**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \ln \pi_\theta(a_t  \vert  s_t) \cdot R_t\right]$$


**With Baseline**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \ln \pi_\theta(a_t  \vert  s_t) \cdot (R_t - b(s_t))\right]$$


### Actor-Critic Methods


**Architecture**:

- **Actor**: Policy network $\pi_\theta(a \vert s)$

- **Critic**: Value network $V_\phi(s)$ or $Q_\phi(s,a)$


**Benefits**:

- Lower variance than pure policy gradient

- More sample efficient than value-based methods

- Can handle continuous action spaces


---

## 7. AlphaGo and Monte Carlo Tree Search

### Overview

AlphaGo is DeepMind's groundbreaking Go-playing AI that defeated world champion players. It combines deep neural networks with Monte Carlo Tree Search (MCTS) for sophisticated game planning.

### Monte Carlo Tree Search (MCTS)

**Key Idea**: Use look-ahead search to evaluate actions by simulating many possible future game trajectories.

**Main Components**:
- Select actions by exploring the game tree
- Look ahead to evaluate whether actions lead to wins or losses
- Repeat simulations many times
- Choose the action with the highest success rate

### MCTS Algorithm Steps

MCTS consists of four iterative steps:

#### Step 1: Selection
- Start from the root node (current state $s_t$)
- Traverse the tree using a selection policy
- Balance exploration and exploitation using UCB (Upper Confidence Bound):
  $$UCB = Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}$$
  where:
  - $Q(s,a)$: Average value of action $a$ in state $s$
  - $N(s)$: Number of times state $s$ was visited
  - $N(s,a)$: Number of times action $a$ was selected in state $s$
  - $c$: Exploration constant

#### Step 2: Expansion
**Question**: What will be the opponent's action?
- Given player action $a_t$, opponent's action $a_t'$ leads to new state $s_{t+1}$
- Opponent's action is randomly sampled from policy:
  $$a_t' \sim \pi(\cdot \vert s_t'; \theta)$$
  where $s_t'$ is the state observed by the opponent

#### Step 3: Evaluation
- Run a rollout to the end of the game (step $T$)
- Player's action: $a_k \sim \pi(\cdot \vert s_k; \theta)$
- Opponent's action: $a_k' \sim \pi(\cdot \vert s_k'; \theta)$
- Receive terminal reward $r_T$:
  - Win: $r_T = +1$
  - Lose: $r_T = -1$
- Evaluate state $s_{t+1}$ using value network:
  $$v(s_{t+1}; \mathbf{w})$$

#### Step 4: Backup
- MCTS repeats simulations many times
- Each child of $a_t$ has multiple recorded $V(s_{t+1})$ values
- Update action-value:
  $$Q(a_t) = \text{mean}(\text{recorded } V\text{ values})$$
- The Q values will be used in Step 1 (selection)

### Decision Making After MCTS

After running MCTS simulations:
- $N(a)$: Number of times action $a$ has been selected
- Final decision:
  $$a_t = \arg\max_a N(a)$$
- Choose the action that was explored most frequently

**Key Properties**:
- For each new action, AlphaGo performs MCTS from scratch
- Initialize $Q(a)$ and $N(a)$ to zeros for each new position

### AlphaGo's Neural Networks

AlphaGo uses two deep neural networks:

#### 1. Policy Network $\pi(a \vert s; \theta)$
- Predicts probability distribution over legal moves
- Trained in three steps:
  1. **Behavior Cloning**: Train on expert human games
  2. **Policy Gradient**: Improve through self-play
  3. **Value Network Training**: Train value function

#### 2. Value Network $v(s; \mathbf{w})$
- Evaluates board positions
- Predicts probability of winning from state $s$
- Used in MCTS evaluation step

### Training Process

**Training in 3 steps**:

1. **Train a policy network using behavior cloning**
   - Learn from expert human games
   - Supervised learning on professional game datasets

2. **Train the policy network using policy gradient algorithm**
   - Self-play reinforcement learning
   - Improve beyond human-level play

3. **Train a value network**
   - Predict game outcomes from board positions
   - Used to evaluate leaf nodes in MCTS

### MCTS Summary

- MCTS has 4 steps: **selection, expansion, evaluation**, and **backup**
- To perform one action, AlphaGo repeats the 4 steps many times to calculate $Q(a)$ and $N(a)$ for every action $a$
- AlphaGo executes the action $a$ with the highest $N(a)$ value
- To perform the next action, AlphaGo performs MCTS all over again (initializing $Q(a)$ and $N(a)$ to zeros)

### Key Innovations

**AlphaGo's Success Factors**:
1. **Combination of Deep Learning and Tree Search**: Neural networks guide MCTS
2. **Self-Play**: Continuous improvement through playing against itself
3. **Value and Policy Networks**: Dual network architecture for position evaluation and move selection
4. **Efficient Search**: MCTS focuses computational resources on promising moves

---


## 8. Continuous Control


### Discrete vs Continuous Control


**Discrete Control**:

- Finite action space (e.g., left, right, up, down)

- Can enumerate all actions

- Use softmax for action probabilities


**Continuous Control**:

- Infinite action space (e.g., steering angle, throttle)

- Cannot enumerate actions

- Need different policy representations


### Policy Representations for Continuous Actions


**Gaussian Policy**:

$$\pi_\theta(a \vert s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$$


where:

- $\mu_\theta(s)$: Mean action (neural network output)

- $\sigma_\theta(s)$: Standard deviation (neural network output or learnable parameter)


**Beta Policy** (for bounded actions):

$$\pi_\theta(a \vert s) = \text{Beta}(\alpha_\theta(s), \beta_\theta(s))$$


### Deterministic Policy Gradient


**Key Insight**: For continuous actions, deterministic policies can be more efficient:

$$\mu_\theta: S \rightarrow A$$


**Policy Gradient**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q(s,a)\Big\vert_{a=\mu_\theta(s)}\right]$$


**Popular Algorithms**:

- **DDPG** (Deep Deterministic Policy Gradient)

- **TD3** (Twin Delayed DDPG)

- **SAC** (Soft Actor-Critic)


---


## 9. Multi-Agent Reinforcement Learning


### MARL Concepts and Challenges


**Multi-Agent Environment**:

- Multiple agents learning simultaneously

- Each agent's environment includes other agents

- Non-stationarity: Environment changes as other agents learn


**Key Challenges**:

1. **Non-stationarity**: Other agents change their policies

2. **Partial Observability**: Agents may not observe other agents' actions

3. **Credit Assignment**: Which agent contributed to team reward?

4. **Coordination**: How to coordinate between agents?

5. **Communication**: Whether and how agents should communicate


### MARL Paradigms


**Independent Learning**:

- Each agent treats others as part of environment

- No coordination or communication

- Simple but can be unstable


**Centralized Training, Decentralized Execution**:

- Training uses global information

- Execution uses only local information

- Popular approach in cooperative settings


**Fully Centralized**:

- Single controller for all agents

- Scalability issues

- Joint action space grows exponentially


### Popular MARL Algorithms


**MADDPG** (Multi-Agent DDPG):

- Centralized critic, decentralized actors

- Each agent has its own actor-critic

- Critics can access global state during training


**QMIX**:

- Value decomposition method

- Individual Q-functions combined into team Q-function

- Monotonicity constraint ensures consistency


---


## Key Equations Summary


### Value Functions

- **Return**: $U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots$

- **Action-Value**: $Q_\pi(s, a) = \mathbb{E}[U_t  \vert  S_t = s, A_t = a]$

- **State-Value**: $V_\pi(s) = \mathbb{E}[U_t  \vert  S_t = s]$


### Learning Updates

- **TD Target**: $y_t = r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})$

- **Q-Learning**: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

- **SARSA**: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$


### Policy Gradient

- **Basic PG**: $\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \ln \pi_\theta(a \vert s) \cdot R\right]$

- **With Baseline**: $\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \ln \pi_\theta(a \vert s) \cdot (R - b)\right]$


### Continuous Control

- **Gaussian Policy**: $\pi_\theta(a \vert s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$

- **Deterministic PG**: $\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q(s,a)\Big\vert_{a=\mu_\theta(s)}\right]$


---


## Practical Implementation Notes


### Common Libraries and Tools

- **OpenAI Gym**: Standard RL environment interface

- **Stable-Baselines3**: High-quality RL algorithm implementations

- **TensorFlow/PyTorch**: Deep learning frameworks

- **Ray RLlib**: Scalable RL library


### Hyperparameter Tuning Tips

- **Learning Rate**: Start with 1e-3 to 1e-4

- **Discount Factor**: Usually 0.9 to 0.999

- **Exploration**: ε-greedy with decay for DQN

- **Network Architecture**: Start simple, add complexity if needed


### Debugging RL Algorithms

1. **Verify Environment**: Test with random policy

2. **Check Shapes**: Ensure tensor dimensions are correct

3. **Monitor Learning**: Plot learning curves

4. **Ablation Studies**: Remove components to understand contributions


---


This comprehensive summary covers the key concepts, algorithms, and mathematical foundations of Deep Reinforcement Learning as presented in Shusen Wang's lecture series. Each section builds upon previous concepts, providing a structured learning path from basic probability theory to advanced multi-agent systems.