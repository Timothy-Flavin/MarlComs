import EscapeEnv
import numpy as np
import random

env = EscapeEnv.env()
initial_observations = env.reset()
print("Map rendered from omnicient viewpoint:")
env.render_full_ascii()
done = False

while not done:
  actions = np.array([[0,0,0,0,0,],[0,0,0,0,0,]])
  # Player 2 takes a random action. move or communicate
  p2_action = random.randint(0,4)
  actions[1,p2_action] = 1

  # Player 1 takes an action prompted by the terminal
  act = input("Player 0 take action ('w','a','s','d'): ")
  if act == 'w':
    actions[0,0] = 1
  if act == 'd':
    actions[0,1] = 1
  if act == 's':
    actions[0,2] = 1
  if act == 'a':
    actions[0,3] = 1
  if act == 'r':
    actions[0,4] = 1

  # Step the environment and get back the needed information
  observations, rewards, done, truncated, info = env.step(actions)
  
  # during some turns, some of the players are dead and therefore their
  # actions are meaningless. This exposes an array of which players took
  # actions where 1 -> did something and 0 -> dead and did nothing. 
  # On the last turn of the environment, where communal rewards are given,
  # took_actions = np.ones(num_players) so that if this is used to mask
  # rewards, it will still supply rewards for the final step when delayed 
  # rewards are given. To discound the rewards by the proper amount do what
  # you want, but you could keep a sum of 1-took_actions and put gamma^sum_arr
  # to get the properly discounted rewards
  took_actions = env.took_actions
  # Print information of interest
  print(f"rewards: {rewards}, done {done}, truncated {truncated}")
  
  # Render for a human from Player 0's perspective
  env.render_full_ascii(playerid=0)

print("Game is over, be sure to play again")