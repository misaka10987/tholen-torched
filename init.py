from os import system
from os.path import exists


def init():
    if not exists("asteroid.db"):
        system("wget https://github.com/misaka10987/tholen-torched/releases/download/data/asteroid.db")
