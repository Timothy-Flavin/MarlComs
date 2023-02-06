import EscapeEnv
import numpy as np

env = EscapeEnv.env()
env.reset()
print("Map rendered from omnicient viewpoint:")
env.render_full_ascii()

for i in range(80):
  actions = np.array([[0,0,0,0,0,],[0,0,0,1,0,]])

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

  print(actions)
  env.step(actions)
  env.render_full_ascii(playerid=0)
  obs1, obs2 = env.obs(id=0)
  print("Observation 1, gens players zombies me")
  print(obs1)
  print("Obervation 2, num alive, gens_left")
  print(obs2)