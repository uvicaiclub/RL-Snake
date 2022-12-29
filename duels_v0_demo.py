from pz_battlesnake.env import duels_v0
import time

env = duels_v0.env()

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
        time.sleep(0.5)
        #env.render() # uncomment this to render
        # This is a shortcut to set the done to be true, 
        # since when all agents are done the env.agents array will be empty
        done = not env.agents