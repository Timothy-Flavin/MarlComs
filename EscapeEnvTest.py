import EscapeEnv
import numpy as np

env = EscapeEnv.env()
env.reset()
print("Map rendered from omnicient viewpoint:")
env.render_full_ascii()

for i in range(30):
  env.step(np.array([5,4,2]))
  env.render_full_ascii(playerid=0)
  obs1, obs2 = env.obs(id=0)
  print("Observation 1, gens players zombies me")
  print(obs1)
  print("Obervation 2, num alive, gens_left")
  print(obs2)