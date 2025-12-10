import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
print(state)
print(info)
print(env.action_space)
action=1
state, reward, terminated, truncated, info = env.step(action)
print(state)
print(reward)
print(terminated)
print(info)