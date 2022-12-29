# First we need to import the enviorment
from pz_battlesnake.env import solo_v0
import time

# Then we can create a new enviorment
#env = solo_v0.env() 

#Create a 15x15 solo enviorment 
env = solo_v0.env(width=15, height=15) # uncomment this to create a 15x15 solo enviorment

# Run for 10 games
for _ in range(10):
    # Reset enviorment, before each game
    env.reset()
    done = False
    while not done:
        for agent in env.agents:
            #print("Agent: ", env.agents)
            # Get Last Observation, Reward, Done, Info
            observation, reward, termination, truncation, info = env.last()
            try:
                print("observation board: ", observation["board"])
            except:
                pass
            # Pick an action, if agenmt is done, pick None
            #print("termination: ", termination)
            if termination:
                action = None
            else:
                action = env.action_space(agent).sample()
            # Step through the enviorment
            env.step(action)
        # Code below runs, when all agents has taken an action
        # Render the enviorment
        #time.sleep(0.5)
        #env.render() # uncomment this to render
        # This is a shortcut to set the done to be true, 
        # since when all agents are done the env.agents array will be empty
        done = not env.agents