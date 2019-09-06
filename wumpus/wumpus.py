#!/usr/bin/env python
from networkx import DiGraph
from pyswip import Prolog

""" Pseudo-Code
    finish = player has 3 pieces of gold and is at the entrance. 
    PS: this implies the player knows the number of gold pieces.
    while player is alive and not finish:
        flag = 0.
        while there is no wumpus (the player can take a safe action):
            if it glitters, get the gold.
            else if the player can move foward to an unvisited safe (ok) square, move foward.
            else if there are safe unvisited squares adjacent to the player, pick one of them and face them.
            else if not flag and there are safe previously visited squares adjacent to the player
            and there are known safe unvisited squares (possibly non-adjacent to the player):
                    pick a safe unvisited square X and Goto(current position, X).
                    if no path exists to X, pick another X, and if no path
                    exists at all, flag = 1.
            else if there is a known wumpus not in a hole and adjacent to the player and the player can shoot:
                face the wumpus.
                Fight().
            else if there is a known wumpus not in a hole adjacent to a safe previously visited square X
            (possibly non-adjacent to the player), and the player can shoot:
                Goto(current position, X). If there is no path, pick another X, if no path exists, break.
                face the wumpus.
                Fight().
            else break.

            if it glitters, get the gold.
            (the player must take an unsafe action)
            if there is a wumpus:
                if the players life is less than or equal to max damage of wumpus:
                    unless there is a known wumpus or hole in the next square, move foward.
                    otherwise, (on the off chance the next hit doesnt kill you due to randomness):
                        pick the closest adjacent safe square X and Backtrack(X).
                else pick the closest adjacent safe square X and Backtrack(X).
            else if there are smelly squares:
                pick an unsafe square X adjacent to at least one, but the smallest number of, smelly squares.
                Goto(current position, X). If there is no path, pick another X, if no path exists, break.




    Backtrack(X):
        face X.
        move foward.
        if there was a wumpus on the previous square, and it glittered:
            (optional) calculate expected value of dropping all remaining arrows into the wumpus and if it is positive:
            turn backward.
            Fight().
    
    Fight():
        while you dont hear a scream and have arrows left, shoot.
        if you hear a scream, move foward.
        else if the square in front glitters and 
        your life is bigger than 3 times the max damage of the wumpus -3:
             move foward.
             grab the gold.
             pick the closest adjacent safe square X and Backtrack(X).

"""


class Agent(DiGraph):
    def __init__(self, knowledge_base=None):
        self.knowledge_base = Prolog()
        self.time = 0
        if knowledge_base:
            self.knowledge_base.consult(knowledge_base)


    def perceive(self, perception):
        pass
