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

class generator: 
  def __init__(self, id, x, y):
    self.id = id
    self.x = x
    self.y = y
    self.completed = 0

class env:
  def __init__(self, map_size = [10,10], n_players = 2, n_zoms=1, gen_locs = np.array([[4,4], [1,8], [8,8]]), player_start_locs = np.array([[8,1], [7,2]]), i_decay = 0.2, ):
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
          if rec > 1.1*self.i_decay:
            rec = rec-self.i_decay
          # if the player knows the gen is complete, the recency stays at 1
          if p.gens_info[g.id,3] > 0:
            rec = 1
          p.gens_info[g.id] = np.array([p.gens_info[g.id,0],p.gens_info[g.id,1], rec, p.gens_info[g.id,3]])

  def update_zombie_info(self):
    for z in self.zombies:
      for p in self.players:
        if self.viewable_by(p.x,p.y,z):
          z.player_locs[p.id] = np.array([p.x,p.y,1.0])
        else:
          z.player_locs[p.id,2] = max(z.player_locs[p.id,2]-self.i_decay,0)
  
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
        if player is not None:
          zmap[-1].append("--")
        else:
          zmap[-1].append("  ")
    for g in self.gens:
      if player is None or self.viewable_by(g.x,g.y, player) or player.gens_info[g.id,3] > 0:
        zmap[g.x][g.y] = f"G{g.id}"
      #TODO make it so that the generator has the number and letter flipped if it is completed
    for p in self.players:
      if player is None or self.viewable_by(p.x,p.y, player):
        zmap[p.x][p.y] = f"P{p.id}"
    for z in self.zombies:
      if player is None or self.viewable_by(z.x,z.y, player):
        zmap[z.x][z.y] = f"Z{z.id}"
    
    if player is not None:
      for y in range(player.y-player.view_range, player.y+player.view_range+1):
        for x in range(player.x-player.view_range, player.x+player.view_range+1):
          if not self.oob(x, y):
            if zmap[x][y] == "--":
              zmap[x][y] = "  "
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

  # out of bounds utility
  def oob(self, x,y):
    if x < 0 or x >= self.map_size[0] or y<0 or y >= self.map_size[1]:
      return True
    else:
      return False

  def obs(self, id, verbose=True):
    """ 
    Returns an observation from the perspective of the
    the player with the given id. view range is how far the 
    player can see in each direction so for view range = x
    the player can see in a x+1+x by x+1+x box 
    
    obs format: np array(4,mapsize[0],mapsize[1])
                (gens, players, zombies, me) 
    where 
    gens: np float array(shape = map_size) 
          0 is a space not occupied by an unfinished 
          gen and > 0 is occupied
    players: same thing but players, 0 is dead
    zombies: same thing but zombies 

    and another array of misc information
    int: players alive from this player's pov
    int: gens left from this players pov     
    binary: door open

    """
    
    player = self.players[id]
    view_range = player.view_range
    x = player.x
    y = player.y
    #if verbose:
      #print(f"observation for player: {id}")
    obs1 = np.zeros(shape=(4, self.map_size[0], self.map_size[1]), dtype=np.float32)
    players_alive = 0
    gens_left = 0

    for g in player.gens_info:
      if g[3] == 0:
        gens_left +=1
        obs1[0,int(g[1]),int(g[0])] = g[2]
    
    for p in player.players_info:
      p_status = min(p[2], p[3])
      if p[3] > 0:
        players_alive +=1
      obs1[1,int(p[1]),int(p[0])] = p_status
    for z in player.zombie_info:
      obs1[2,int(z[1]),int(z[0])] = z[2]
    obs1[3,int(x),int(y)] = 1

    obs2 = np.array([players_alive, gens_left])

    return obs1, obs2
  
  def complete_gen(self, x, y):
    reward = 0
    for g in self.gens:
      if x == g.x and y == g.y and g.completed != 1:
        print("repaired gen!")
        g.completed=1
        self.gens_active -= 1
        reward+=1
    return reward
    
  def player_move(self, id, dir):
    reward = 0
    #if player is dead then don't move or reward them
    if self.players[id].alive == 0:
      print(f"player {id} dead")
      return reward
    x = self.players[id].x
    y = self.players[id].y
    if dir == 0:
      y-=1
    elif dir==1:
      x+=1
    elif dir==2:
      y+=1
    elif dir==3:
      x-=1
    
    for z in self.zombies:
      if x==z.x and y==z.y:
        self.players[id].alive = 0
        # remove self from zombie's list
        z.player_locs[id,2] = 0
        #self.players[id].num_alive -=1
        reward -=1
        return reward

    if not self.oob(x,y):
      self.players[id].x = x
      self.players[id].y = y
      if self.gens_active == 0 and x==0 and y==0:
        self.players[id].alive=0
        reward += 5
    return reward

  def zombie_move(self, z):
    max_dist = self.map_size[0] * self.map_size[1]
    target = - 1    
    for p in range(z.player_locs.shape[0]):
      print(z.player_locs[p])
      if z.player_locs[p,2] > self.i_decay/2:
        p_dist = abs(z.x - z.player_locs[p,0]) + abs(z.y - z.player_locs[p,1])
        print(f"max dist {max_dist}, pdist {p_dist}")
        if p_dist < max_dist and self.players[p].alive>0:
          max_dist = p_dist
          target = p
          if p_dist == 0:
            self.players[p].alive=False
            z.player_locs[p,2] = 0
    
    if target == -1:
      dx = random.randint(-1,1)
      dy = 0
      if dx == 0:
        dy = random.randint(-1,1)
      if not self.oob(z.x + dx, z.y + dy):
        z.x = z.x+dx
        z.y = z.y+dy
    else:
      print(f"Target player {target}")
      dx = z.player_locs[target,0] - z.x
      dy = z.player_locs[target,1] - z.y
      if dx != 0 and dy != 0:
        choice = random.randint(0,1)
        if choice == 0:
          if dx>0:
            z.x+=1
          if dx<0:
            z.x-=1
        else:
          if dy>0:
            z.y+=1
          if dy<0:
            z.y-=1
      else:
        if dx>0:
          z.x+=1
        if dx<0:
          z.x-=1
        if dy>0:
          z.y+=1
        if dy<0:
          z.y-=1
    
  def step(self, actions, verbose=False):
    rewards = np.zeros(actions.shape[0])
    if verbose==True:
      for agent in range(actions.shape[0]):
        print(f"Agent {agent} Took action: {actions[agent]}")
        
    for agent in range(actions.shape[0]):
      act = np.argmax(actions[agent])
      print(f"act {act}")
      if act<4:
        rewards[agent] += self.player_move(agent,act)
      elif act == 4:
        self.complete_gen(self.players[agent].x, self.players[agent].y)
    
    for z in self.zombies:
      self.zombie_move(z)
      

    self.update_zombie_info()
    self.update_player_info()
    

