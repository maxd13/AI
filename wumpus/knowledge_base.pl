%conhecimento basico.

bounded(I, J) :- 
    I >= 1,
    J >= 1,
    I =< 12,
    J =< 12.

%ordenado por prioridade.
action(get_gold).
action(move).
action(turn_left).
action(turn_right).
action(shoot).

direction(left).
direction(right).
direction(up).
direction(down).

:- dynamic smells/2, breezes/2, glitter/2, ok/2, take/2, damage/2, dead/2.
%take(Action, Time) = make-action-sentence.

%visited(1,1).

%tudo o que reluz e' ouro.
gold(I, J) :- glitter(I, J).

ok(1, 1).
%util para evitar verificacoes de bounded em outras regras
ok(I, J)   :- not(bounded(I, J)).
ok(I, J+1) :- not(smells(I, J)), not(breezes(I, J)).
ok(I, J-1) :- not(smells(I, J)), not(breezes(I, J)).
ok(I+1, J) :- not(smells(I, J)), not(breezes(I, J)).
ok(I-1, J) :- not(smells(I, J)), not(breezes(I, J)).
ok(I, J)   :- not(wumpus(I, J)), not(hole(I, J)).

take(get_gold, Time) :-
    player(Player, Time),
    cantake(get_gold, Player).

take(Action, Time) :-
    action(Action),
    player(Player, Time),
    cantake(Action, Player),
    action(Other),
    Other \= Action,
    not(cantake(Other, Player)).

life(T, L) :- player([_, _, _, _, L], T).

Cost = 1.
SCost = 10.
Gain = 1000.
player(X, 0) :- X = [1, 1, 5, right, 100].
player(X, T) :- 
    player([I, J, Shots, Facing, Life], T - 1),
    take(get_gold, T - 1),
    damage(D, T),
    X = [I, J, Shots, Facing, Life - damage(X, T) + Gain - Cost].
%move
player(X, T) :- 
    player([I, J, Shots, right, Life], T - 1),
    take(move, T - 1),
    X = [I, J+1, Shots, right, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, left, Life], T - 1),
    take(move, T - 1),
    X = [I, J-1, Shots, left, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, up, Life], T - 1),
    take(move, T - 1),
    X = [I+1, J, Shots, up, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, down, Life], T - 1),
    take(move, T - 1),
    X = [I+1, J, Shots, down, Life - damage(X, T) - Cost].
%turn left
player(X, T) :- 
    player([I, J, Shots, right, Life], T - 1),
    take(turn_left, T - 1),
    X = [I, J, Shots, up, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, up, Life], T - 1),
    take(turn_left, T - 1),
    X = [I, J, Shots, left, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, left, Life], T - 1),
    take(turn_left, T - 1),
    X = [I, J, Shots, down, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, down, Life], T - 1),
    take(turn_left, T - 1),
    X = [I, J, Shots, right, Life - damage(X, T) - Cost].
%turn right
player(X, T) :- 
    player([I, J, Shots, right, Life], T - 1),
    take(turn_right, T - 1),
    X = [I, J, Shots, down, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, up, Life], T - 1),
    take(turn_right, T - 1),
    X = [I, J, Shots, right, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, left, Life], T - 1),
    take(turn_right, T - 1),
    X = [I, J, Shots, up, Life - damage(X, T) - Cost].
player(X, T) :- 
    player([I, J, Shots, down, Life], T - 1),
    take(turn_right, T - 1),
    X = [I, J, Shots, left, Life - damage(X, T) - Cost].
%shoot
player(X, T) :- 
    player([I, J, Shots, Facing, Life], T - 1),
    take(shoot, T - 1),
    X = [I, J, Shots-1, Facing, Life - damage(X, T) - SCost].



cantake(get_gold, Player) :-
    Player = [I, J, _, _, _],
    gold(I, J).

cantake(move, Player) :-
    not(cantake(get_gold, Player)),
    Player = [I, J, _, Facing, _],
    Facing = right,
    bounded(I, J+1),
    ok(I, J+1).

cantake(move, Player) :-
    not(cantake(get_gold, Player)),
    Player = [I, J, _, Facing, _],
    Facing = left,
    bounded(I, J-1),
    ok(I, J-1).

cantake(move, Player) :-
    not(cantake(get_gold, Player)),
    Player = [I, J, _, Facing, _],
    Facing = up,
    bounded(I+1, J),
    ok(I+1, J).

cantake(move, Player) :-
    not(cantake(get_gold, Player)),
    Player = [I, J, _, Facing, _],
    Facing = down,
    bounded(I-1, J),
    ok(I-1, J).

cantake(turn_left, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,right,_],
    (cantake(move, [I,J,_,up,_]) ; cantake(turn_left, [I,J,_,up,_])).

cantake(turn_left, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,up,_],
    (cantake(move, [I,J,_,left,_]) ; cantake(turn_left, [I,J,_,left,_])).

cantake(turn_left, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,left,_],
    (cantake(move, [I,J,_,down,_]) ; cantake(turn_left, [I,J,_,down,_])).

cantake(turn_left, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,down,_],
    (cantake(move, [I,J,_,right,_]) ; cantake(turn_left, [I,J,_,right,_])).

cantake(turn_right, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,right,_],
    (cantake(move, [I,J,_,down,_]) ; cantake(turn_right, [I,J,_,down,_])).

cantake(turn_right, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,up,_],
    (cantake(move, [I,J,_,right,_]) ; cantake(turn_right, [I,J,_,right,_])).

cantake(turn_right, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,left,_],
    (cantake(move, [I,J,_,up,_]) ; cantake(turn_right, [I,J,_,up,_])).

cantake(turn_right, Player) :-
    not(cantake(get_gold, Player)),
    not(cantake(move, Player)),
    Player = [I,J,_,down,_],
    (cantake(move, [I,J,_,left,_]) ; cantake(turn_right, [I,J,_,left,_])).

cantake(shoot, Player) :-
    action(X),
    X \= shoot,
    not(cantake(X, Player)),
    Player = [I, J, Shots, right , _],
    Shots > 0,
    wumpus(I, Y),
    Y > J.

cantake(shoot, Player) :-
    action(X),
    X \= shoot,
    not(cantake(X, Player)),
    Player = [I, J, Shots, up , _],
    Shots > 0,
    wumpus(Y, J),
    Y > I.

cantake(shoot, Player) :-
    action(X),
    X \= shoot,
    not(cantake(X, Player)),
    Player = [I, J, Shots, left , _],
    Shots > 0,
    wumpus(I, Y),
    Y < J.

cantake(shoot, Player) :-
    action(X),
    X \= shoot,
    not(cantake(X, Player)),
    Player = [I, J, Shots, down , _],
    Shots > 0,
    wumpus(Y, J),
    Y < I.

%regras de inferencia.

wumpus(I, J) :-
    bounded(I, J),
    bounded(I+1, J),
    not(dead(I,J)),
    smells(I+1, J),
    not(wumpus(I+1, J)),
    not(wumpus(I+1, J-1)),
    not(wumpus(I+1, J+1)),
    not(wumpus(I+2, J)).

wumpus(I, J) :-
    bounded(I, J),
    bounded(I-1, J),
    not(dead(I,J)),
    smells(I-1, J),
    not(wumpus(I-1, J)),
    not(wumpus(I-1, J-1)),
    not(wumpus(I-1, J+1)),
    not(wumpus(I-2, J)).

wumpus(I, J) :-
    bounded(I, J),
    bounded(I, J+1),
    not(dead(I,J)),
    smells(I, J+1),
    not(wumpus(I, J+1)),
    not(wumpus(I-1, J+1)),
    not(wumpus(I+1, J+1)),
    not(wumpus(I, J+2)).

wumpus(I, J) :-
    bounded(I, J),
    bounded(I, J-1),
    not(dead(I,J)),
    smells(I, J-1),
    not(wumpus(I, J-1)),
    not(wumpus(I-1, J-1)),
    not(wumpus(I+1, J-1)),
    not(wumpus(I, J-2)).

hole(I, J) :-
    bounded(I, J),
    bounded(I+1, J),
    breezes(I+1, J),
    not(hole(I+1, J)),
    not(hole(I+1, J-1)),
    not(hole(I+1, J+1)),
    not(hole(I+2, J)).

hole(I, J) :-
    bounded(I, J),
    bounded(I-1, J),
    breezes(I-1, J),
    not(hole(I-1, J)),
    not(hole(I-1, J-1)),
    not(hole(I-1, J+1)),
    not(hole(I-2, J)).

hole(I, J) :-
    bounded(I, J),
    bounded(I, J+1),
    breezes(I, J+1),
    not(hole(I, J+1)),
    not(hole(I-1, J+1)),
    not(hole(I+1, J+1)),
    not(hole(I, J+2)).

hole(I, J) :-
    bounded(I, J),
    bounded(I, J-1),
    breezes(I, J-1),
    not(hole(I, J-1)),
    not(hole(I-1, J-1)),
    not(hole(I+1, J-1)),
    not(hole(I, J-2)).