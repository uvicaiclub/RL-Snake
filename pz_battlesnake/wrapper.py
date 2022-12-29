import ctypes
import json
import os

# Load the shared library from the proper path
here = os.path.abspath(os.path.dirname(__file__))
file = "/home/nathanwoloshyn/Desktop/projects/ML/battlesnake_arena/pz-battlesnake/build/lib.linux-x86_64-cpython-39/bin/battlesnake" # HACK TO FIX WRONG PATH, REPLACE with you path to the built battlesnake files
#print(file)
if os.name == "nt":
    battlesnake = ctypes.CDLL(file)
elif os.name == "posix":
    battlesnake = ctypes.CDLL(file)
else:
    raise Exception("Unsupported OS")

# Setup
_setup = battlesnake.setup
_setup.argtypes = [ctypes.c_char_p]

# Reset
_reset = battlesnake.reset
_reset.argtypes = [ctypes.c_char_p]
_reset.restype = ctypes.c_char_p

# step
_step = battlesnake.step
_step.argtypes = [ctypes.c_char_p]
_step.restype = ctypes.c_char_p

# isGameOver
_done = battlesnake.isGameOver
_done.restype = ctypes.c_int

# render
_render = battlesnake.render
_render.argtypes = [ctypes.c_int]


def env_render(color: bool = True):
    _render(1 if color else 0)


def env_done():
    return _done() == 1


def env_setup(options: dict):
    # Convert options to string
    options = json.dumps(options).encode("utf-8")

    # Call setup in go
    _setup(options)


def env_reset(options: dict):
    # Convert options to string
    options = json.dumps(options).encode("utf-8")

    # Call reset in go
    res = _reset(options)

    return json.loads(res.decode("utf-8"))

def int_to_action(actions):
    action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
    for agent in actions.keys():
        actions[agent] = action_dict[actions[agent]]
    return actions

def env_step(actions: dict):
    actions = int_to_action(actions)
   # print(actions)

    # Convert actions to string
    actions = json.dumps(actions).encode("utf-8")

    # Call step in go
    res = _step(actions)

    # convert result to python object
    res = json.loads(res.decode("utf-8"))

    # return result
    return res
