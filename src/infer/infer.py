import os

def server_central():
    print("IM CENTRAL")

def server_expert():
    print("IM EXPERT")

def start():
    type = os.getenv("type")

    if (type == "central"):
        server_central()
    else:
        server_expert()

start()