import numpy as np

class GridWorld:
    def __init__(self):
        self.n_states = 16
        self.n_actions = 4
        self.gamma = 0.9
        self.P = {}
        for s in range(self.n_states):
            self.P[s] = {a: [] for a in range(self.n_actions)}
        for i in range(4):
            for j in range(4):
                s = 4*i + j
                for a in range(self.n_actions):
                    next_s = s
                    if a == 0:
                        next_i = i-1
                        next_j = j
                    elif a == 1:
                        next_i = i
                        next_j = j+1
                    elif a == 2:
                        next_i = i+1
                        next_j = j
                    elif a == 3:
                        next_i = i
                        next_j = j-1
                    if next_i >= 0 and next_i <= 3 and next_j >= 0 and next_j <= 3:
                        next_s = 4*next_i + next_j
                    if s == 0 or s == 15:
                        self.P[s][a].append((1.0, s, 0, True))
                    else:
                        self.P[s][a].append((1.0, next_s, -1, False))


class PolicyEvaluation:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.n_states)
        # self.policy = np.zeros(env.n_states, dtype=int)
            
    def train(self):
        while True:
            delta = 0
            for s in range(self.env.n_states):
                v = self.V[s]
                new_v = 0
                for a in range(self.env.n_actions):
                    for prob, next_s, reward, done in self.env.P[s][a]:
                        new_v += 0.25 * (reward + self.env.gamma * self.V[next_s])
                self.V[s] = new_v
                delta = max(delta, abs(v - self.V[s]))
            if delta < 1e-6:
                break
        print("Value Function:")
        print(np.reshape(np.round(self.V, decimals=3), (4, 4)))
        
        
        
class PolicyIteration:
    def __init__(self, env):  
        self.env = env
        
    def policy_evaluation(self, policy, theta=0.0001):
        V = np.zeros(self.env.n_states)
        while True:
            delta = 0
            for s in range(self.env.n_states):
                v = V[s]
                v_new = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v_new += action_prob * prob * (reward + self.env.gamma * V[next_state])
                V[s] = v_new
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V
    
    def policy_improvement(self, V):
        policy = np.zeros([self.env.n_states, self.env.n_actions]) / self.env.n_actions
        for s in range(self.env.n_states):
            q_values = np.zeros(self.env.n_actions)
            for a in range(self.env.n_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    q_values[a] += prob * (reward + self.env.gamma * V[next_state])
            best_a = np.argmax(q_values)
            policy[s, best_a] = 1.0
        return policy
    
    def train(self):
        policy = np.ones([self.env.n_states, self.env.n_actions]) / self.env.n_actions
        while True:
            V = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(V)
            if np.all(policy == new_policy):
                break
            policy = new_policy
        print("Value Function:")
        print(np.reshape(V, (4, 4)))
        print("Optimal Policy:")
        policy_max = np.argmax(policy, axis=1)
        arrows = np.array(["^", ">", "v", "<"])[policy_max]
        grid = np.reshape(arrows, (4, 4))
        print(grid)



class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)
        
    def train(self):
        step = 0
        while True:
            step += 1
            delta = 0
            for s in range(self.env.n_states):
                v = self.V[s]
                action_values = np.zeros(self.env.n_actions)
                for a in range(self.env.n_actions):
                    next_s, reward, _ = self.env.P[s][a][0][1:]
                    action_values[a] = reward + self.env.gamma * self.V[next_s]
                self.V[s] = np.max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            if delta < 1e-6:
                break
                
        for s in range(self.env.n_states):
            action_values = np.zeros(self.env.n_actions)
            for a in range(self.env.n_actions):
                next_s, reward, _ = self.env.P[s][a][0][1:]
                action_values[a] = reward + self.env.gamma * self.V[next_s]
            self.policy[s] = np.argmax(action_values)
        print('{} steps to convergence'.format(step))
        print("Value Function:")
        print(np.reshape(self.V, (4, 4)))
        print("Optimal Policy:")
        arrows = np.array(["^", ">", "v", "<"])[self.policy]
        grid = np.reshape(arrows, (4, 4))
        print(grid)




        
# print('Policy Evaluation Algorithem')
# env3 = GridWorld()
# agent3 = PolicyEvaluation(env3)
# agent3.train()

# print('Policy Iteration Algorithem')
# env2 = GridWorld()
# agent2 = PolicyIteration(env2)
# agent2.train()

print('Value Iteration Algorithem')
env = GridWorld()
agent = ValueIteration(env)
agent.train()
