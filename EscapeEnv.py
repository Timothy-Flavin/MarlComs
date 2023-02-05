import numpy as np
import random

class survivor:
  def __init__(self, id, x, y, n_gens, n_players, n_zoms, view_range = 2):
    self.id = id
    self.x = x
    self.y = y
    self.alive = True
    
    self.gens_info = np.zeros(shape=(n_gens,4)) # [gen id] [dx, dy, recency, completed]
    # gen dx = 0 dy = 0 and recency is -1 if not seen or completed and completed is 0 or 1
    self.players_info = np.zeros(shape=(n_players,4)) # [player id] [dx, dy, recency, alive]
    self.players_info[:,3] = 1
    print(f"player {id}'s player info:")
    print(self.players_info)
    self.zoms_info = np.zeros(shape=(n_players,3)) # [zom id] [dx, dy, recency]
    self.view_range = view_range

class zombie:
  def __init__(self, id, x, y, view_range = 2):
    self.id = id
    self.x = x
    self.y = y
    self.view_range = view_range

  def move():
    print("zombie moves")

class generator: 
  def __init__(self, id, x, y):
    self.id = id
    self.x = x
    self.y = y
    self.completed = False


class env:
  def __init__(self, map_size = [10,10], n_players = 2, n_zoms=1, gen_locs = np.array([[4,4], [1,8], [8,8]]), player_start_locs = np.array([[8,1], [7,2]])):
    self.n_players = n_players
    self.player_ids = np.arange(n_players)
    print(f"Player id's {self.player_ids}")
    self.n_zoms = n_zoms
    self.map_size = np.array(map_size)
    self.gen_locs = gen_locs
    self.n_gens = len(gen_locs)
    self.player_start_locs = player_start_locs

  def reset(self, randomize_gens = True):
    if randomize_gens:
      genlocs = np.random.choice(a=np.arange(self.map_size[0]*self.map_size[1]),size=self.n_gens, replace=False)
      print("Randomizing generator locations")
      print(genlocs)
      for i in range(self.n_gens):
        self.gen_locs[i] = np.array([genlocs[i]%self.map_size[0], int(genlocs[i] / self.map_size[1])])
      print(self.gen_locs)

    self.gens_active = self.n_gens
    self.door_open = False
    
    self.players = []
    for i in range(len(self.player_start_locs)):
      self.players.append(survivor(id=i, x=self.player_start_locs[i,0], y=self.player_start_locs[i,1], n_gens = self.n_gens, n_players = self.n_players, n_zoms = self.n_zoms))
    self.gens = []
    for i in range(self.n_gens):
      self.gens.append(generator(i,self.gen_locs[i,0], self.gen_locs[i,1]))
    self.zombies = []
    for i in range(self.n_zoms):
      onplayer = True
      x,y = 0,0
      while onplayer:
        onplayer = False
        x = random.randint(0,self.map_size[0]-1)
        y = random.randint(0,self.map_size[1]-1)
        for p in self.players:
          print(f"x: {x}, y: {y}, px: {p.x}, py:{p.y}")
          if x == p.x and y == p.y:
            onplayer = True
      self.zombies.append(zombie(i,x,y,3))


  def render_full_ascii(self, playerid=None):
    player=None
    if playerid is not None:
      player = self.players[playerid]
    zmap = []
    for y in range(self.map_size[1]):
      zmap.append([])
      for x in range(self.map_size[0]):
        zmap[-1].append("  ")
    for g in self.gens:
      if player is None or (abs(player.x-g.x)<=player.view_range and abs(player.y-g.y)<=player.view_range):
        zmap[g.x][g.y] = f"g{g.id}"
    for p in self.players:
      if player is None or (abs(player.x-p.x)<=player.view_range and abs(player.y-p.y)<=player.view_range):
        zmap[p.x][p.y] = f"p{p.id}"
    for z in self.zombies:
      if player is None or (abs(player.x-z.x)<=player.view_range and abs(player.y-z.y)<=player.view_range):
        zmap[z.x][z.y] = f"z{z.id}"
    
    for i in range(self.map_size[0]+2):
      print("**", end="")
    print("")
    for y in range(self.map_size[1]):
      print("**",end="")
      for x in range(self.map_size[0]):
        print(zmap[x][y],end="")
      print("**")
    for i in range(self.map_size[0]+2):
      print("**", end="")
    print("")


  def oob(self, x,y):
    if x < 0 or x >= self.map_size[0] or y<0 or y >= self.map_size[1]:
      return False
    else:
      return True

  def obs(self, player, view_range = 2):
    print(f"observation for player: {player.id}")
    

    for y in range(player.y-view_range, player.y+view_range+1):
      for x in range(player.x-view_range, player.x+view_range+1):
        print("hi")


  def step(self, actions, verbose=True):
    if verbose==True:
      for agent in list(actions.keys()):
        print(f"Agent {agent} Took action: {actions[agent]}")

    

