import EscapeEnv
import numpy as np

env = EscapeEnv.env()
env.reset()
env.render_full_ascii()
#env.render_full_ascii(playerid=0)
#env.render_full_ascii(playerid=1)

#env.step(np.array([5,4,2]))
#env.render_full_ascii()
#env.render_full_ascii(playerid=0)
#env.render_full_ascii(playerid=1)

for i in range(20):
  env.step(np.array([5,4,2]))
  env.render_full_ascii(playerid=0)