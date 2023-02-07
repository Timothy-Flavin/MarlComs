# MarlComs
MarlComs is a multi-agent reinforcement learning explainability project focusing on communication among agents.

## Environment Description:

The environment shown below consists of a number of survivor agents and a number of zombie agents. 
- The agents which are the targets for reinforcement learning are the survivors, denoted by 'P#' or 'p#' in the display below. Survivors have the goal of activating several generators. 
- Generators denoted 'G#', or 'g#', do not move and they must all be activated before survivors can exit the game by arriving in the top left square. Generators can be repaired by survivors standing on top of them. The repair action takes up a *number of turns instead of movement.
- One or more Zombies, denoted by 'Z#' or 'z#' will pursue the survivors if they can see them. they can knock out survivors by occupying the same square as the survivor. Zombies have their own view range and they will chase the nearest survivor. 
- Both survivors and zombies keep a memory of the last known location of entities of interest. This memory is reprisented by the lower-case version of that entity when rendered and it will remain in the last seen location of that entity. After several turns, mobile entities (survivors and zombies) will dissapear completely as their last known location becomes less and less reliable. 
- *Survivors can call out information to eachother about important entity locations. For instance a zombie call out from survivor 0 will alert survivor 1 of the location of any zombie that survivor 0 zero can see, but survivor 0's location will also be added to every zombie's internal memory.
- below, one can see an example game being played from survivor 0's perspective with the initial frame of the video serving to show the entire state of the environment before rendering from the survivor's perspective. 


- ![zombgif](zombAI.gif)

- Rewards will be given to individual survivors on the turn that they complete generators or escape, and half of the rewards that each survivor gained will be added to the other survivor's rewards on the last turn of the game.
- the game ends when all of the survivors have either died or escaped. 
- Because the survivor's view range is only 2 in this example, they do not have enough distance between themselves and the zombie to fix all three generators without player 1's help. 


## Using the environment

Actions for each player are reprisented as a 1-D numpy-array where the argmax is the action taken. The actions are as *follows: [up,right,down,left,repair] so two survivors actions where one wants to go up and another wants to go left would be

```actions = np.array([[1,0,0,0,0,],[0,0,0,1,0,]])```

A typical environment instantiation and step follows the structure of the PettingZoo API https://github.com/Farama-Foundation/PettingZoo but the entire API is not implemented and this project has no affiliation with PettingZoo besides using their standardized format for common opperations.

```
import EscapeEnv
import numpy as np

env = EscapeEnv.env()
# sets up the environment
env.reset()
# renders from omnicient point of view
env.render_full_ascii()

while True:
  actions = np.array([[1,0,0,0,0,],[0,0,0,1,0,]])
  
  observations, rewards, termination, truncation, info = env.step(actions)
  
  # renders from player 1's point of view
  env.render_full_ascii(playerid=0)
  obs1, obs2 = env.obs(id=0)
```

`obs1` is a three dimensional array with the following dimensions: `[entity_type, y, x]` where the entity type is either gens, players, zombies, or my location. The value at a given entry reprisents the location and information recency of that object where 1 is to actively see the object and 0 is not present or no information. The rate at which information decays is an environment parameter, `i_decay [0 to 1]`. 

`EscapeEnv` is the environment file and `EscapeEnvTest` is a simple playable implementation from player 0's perspective.

## TODO

Currently, the order at which information such as player death is computed is not quite correct and generator repair only takes one turn. Additionally, the rewards structure is not fully implemented yet and needs to be reworked to better reprisent the value of the state. 

