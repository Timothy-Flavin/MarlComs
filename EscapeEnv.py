import numpy as np
import random

class survivor:
  def __init__(self, id, x, y, n_gens, n_players, n_zoms, view_range = 2):
    self.id = id
    self.x = x
    self.y = y
    self.alive = 1
    # x and y are saved, but observation will return dx and dy on a scale of [0,1]
    self.gens_info = np.zeros(shape=(n_gens,4)) # [gen id] [x, y, recency, completed]
    # gen dx = 0 dy = 0 and recency is -1 if not seen or completed and completed is 0 or 1
    self.players_info = np.zeros(shape=(n_players,4)) # [player id] [x, y, recency, alive]
    self.players_info[:,3] = 1
    self.zombie_info = np.zeros(shape=(n_players,3)) # [zom id] [x, y, recency]
    self.view_range = view_range

  def __str__(self):
    st = f"Player [{self.id}]: x: {self.x}, y: {self.y}, alive: {self.alive}, view_range: {self.view_range}\n"
    st +="\n"
    for i in self.players_info.shape[0]:
      p = self.players_info[i]
      st += f"  player [{i}] info: x: {p[0]}, y: {p[1]}, recency: {p[2]}, alive: {p[3]}\n"
    st+="\n"
    for i in self.zombie_info.shape[0]:
      z = self.zombie_info[i]
      st += f"  zombie [{i}] info: x: {z[0]}, y: {z[1]}, recency: {z[2]}\n"
    st +="\n"
    for i in self.gens_info.shape[0]:
      g = self.gens_info[i]
      st += f"  gen [{i}] info: x: {g[0]}, y: {g[1]}, recency: {g[2]}, completed: {g[3]}\n"
    st +="\n"

class zombie:
  def __init__(self, id, x, y, view_range = 2, n_players=2):
    self.id = id
    self.x = x
    self.y = y
    self.view_range = view_range
    # recency is an int where zero means the zombie can't see that player
    # the int is decrimented towards zero each turn and the zombie can
    # only act if the int is greater than zero. x,y will not update unless
    # a player is in the zombie's vision or if they communicate
    self.player_locs = np.zeros((n_players, 3)) # player [x, y, recency]

  def move():
    print("zombie moves")

class generator: 
  def __init__(self, id, x, y):
    self.id = id
    self.x = x
    self.y = y
    self.completed = 0

class env:
  def __init__(self, map_size = [10,10], n_players = 2, n_zoms=1, gen_locs = np.array([[4,4], [1,8], [8,8]]), player_start_locs = np.array([[8,1], [7,2]]), i_decay = 0.2):
    self.n_players = n_players
    self.player_ids = np.arange(n_players)
    #print(f"Player id's {self.player_ids}")
    self.n_zoms = n_zoms
    self.map_size = np.array(map_size)
    self.gen_locs = gen_locs
    self.n_gens = len(gen_locs)
    self.player_start_locs = player_start_locs
    self.i_decay = i_decay

  def viewable_by(self, x,y, e2):
    """Returns True if entity x,y is within viewrange of entity e2"""
    return (abs(x-e2.x)<=e2.view_range and abs(y-e2.y)<=e2.view_range)

  def update_player_info(self):
    # set player's initial info
    for p in self.players:
      # if the player can see an entity, set the x and y and recency to 1
      # otherwise, decay the recency
      for z in self.zombies:
        if self.viewable_by(z.x,z.y, p):
          p.zombie_info[z.id] = np.array([z.x,z.y,1.0])
        else:
          p.zombie_info[z.id] = np.array([p.zombie_info[z.id,0],p.zombie_info[z.id,1],max(p.zombie_info[z.id,2]-self.i_decay,0)])

      for op in self.players:
        if self.viewable_by(op.x,op.y,p):
          p.players_info[op.id] = np.array([op.x,op.y,1.0,op.alive])
        else:
          p.players_info[op.id] = np.array([p.players_info[op.id,0],p.players_info[op.id,1],max(p.players_info[op.id,2]-self.i_decay,0), p.players_info[op.id,3]])

      for g in self.gens:
        if self.viewable_by(g.x,g.y,p):
          p.gens_info[g.id] = np.array([g.x,g.y,1.0,g.completed])
        else:
          rec = p.gens_info[g.id,2]
          if rec > self.i_decay:
            rec = rec-self.i_decay
          # if the player knows the gen is complete, the recency stays at 1
          if p.gens_info[g.id,3] > 0:
            rec = 1
          p.gens_info[g.id] = np.array([p.gens_info[g.id,0],p.gens_info[g.id,1], rec, p.gens_info[g.id,3]])


  
  
  def reset(self, randomize_gens = True):
    if randomize_gens:
      genlocs = np.random.choice(a=np.arange(self.map_size[0]*self.map_size[1]),size=self.n_gens, replace=False)
      #print("Randomizing generator locations")
      #print(genlocs)
      for i in range(self.n_gens):
        self.gen_locs[i] = np.array([genlocs[i]%self.map_size[0], int(genlocs[i] / self.map_size[1])])
      #print(self.gen_locs)

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
          if x == p.x and y == p.y:
            onplayer = True
      self.zombies.append(zombie(i,x,y,3))
    self.update_player_info()

  def render_full_ascii(self, playerid=None):
    """Renders the state of the game in ascii. 
       If playerid is none, the entire state is rendered with full information
       If payerid is not none, the state is rendered from that player's pov"""
    player=None
    if playerid is not None:
      player = self.players[playerid]
    zmap = []
    for y in range(self.map_size[1]):
      zmap.append([])
      for x in range(self.map_size[0]):
        zmap[-1].append("  ")
    for g in self.gens:
      if player is None or self.viewable_by(g.x,g.y, player) or g.completed > 0:
        zmap[g.x][g.y] = f"G{g.id}"
    for p in self.players:
      if player is None or self.viewable_by(p.x,p.y, player):
        zmap[p.x][p.y] = f"P{p.id}"
    for z in self.zombies:
      if player is None or self.viewable_by(z.x,z.y, player):
        zmap[z.x][z.y] = f"Z{z.id}"
    
    if player is not None:
      for p in range(player.players_info.shape[0]):
        p_inf = player.players_info[p]
        # if a player has been seen but is not currently being seen render
        # their last location with a lower-case p
        if p_inf[2] > 0 and p_inf[2] < 1:
          zmap[int(p_inf[0])][int(p_inf[1])] = f"p{p}"
        
      for z in range(player.zombie_info.shape[0]):
        z_inf = player.zombie_info[z]
        if z_inf[2]>0 and z_inf[2]<1:
          zmap[int(z_inf[0])][int(z_inf[1])] = f"z{z}"
      
      for g in range(player.gens_info.shape[0]):
        g_inf = player.gens_info[g]
        if g_inf[2]>0 and g_inf[2]<1:
          zmap[int(g_inf[0])][int(g_inf[1])] = f"g{g}"

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


  def step(self, actions, verbose=False):
    if verbose==True:
      for agent in range(actions.shape[0]):
        print(f"Agent {agent} Took action: {actions[agent]}")

    act = input("Player 0 take action ('w','a','s','d'): ")
    if act == 'w':
      self.players[0].y-=1
    if act == 's':
      self.players[0].y+=1
    if act == 'a':
      self.players[0].x-=1
    if act == 'd':
      self.players[0].x+=1
    self.update_player_info()
    
