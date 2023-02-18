from human_preference import Human_Preference
from policy import PPO
from collections import namedtuple
import gym
import torch

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    seed = 1
    torch.manual_seed(seed)
    env.seed(seed)
    
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    hp_model = Human_Preference(observation_space, action_space)
    agent = PPO(observation_space, action_space)

    episodes = 200
    gamma = 0.99
    for i_epoch in range(episodes):
        state = env.reset()
        trajectory = []
        step = 0
        total_reward = 0
        while True:
            step += 1
            action, action_prob = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            reward = hp_model.predict(next_state)[0][0]
            trans = Transition(state, action, action_prob, reward, next_state)
            trajectory.append([state, next_state, action, done])
            agent.store_transition(trans)
            state = next_state
            total_reward += reward
            if done :
                hp_model.add_trajectory(ENV_NAME, trajectory)
                agent.add_score(step)
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch, gamma)
                print('Episode: {}, Score: {}, Total reward:{}'.format(i_epoch+1, step, total_reward))
                break
            
        if (i_epoch + 1) % 4 == 0:
            hp_model.ask_human_pref()
            hp_model.train_model()

    hp_model.plot()
    agent.plot_score()
