disk(1..H) :- disks(H).
larger(D1,D2) :- disk(D1), disk(D2), D1 > D2.
smaller(D1,D2) :- disk(D1), disk(D2), D1 < D2.

moves(2**H-1) :- disks(H).
turn(1..T) :- moves(T).

peg("A";"B";"C").

% on peg: Turn, Peg, Disk
on(0,"A",D) :- disk(D).
goal("C").

1 { move(T,P1,D) : peg(P1), disk(D), on(T-1,P2,D), P1 != P2 } 1 :- turn(T).
on(T,P,D) :- move(T,P,D).
on(T,P,D) :- on(T-1,P,D), turn(T), not move(T,_,D).

:- move(T,_,D1), on(T-1,P,D1), on(T-1,P,D2), D2 < D1.
:- move(T,P,D1), on(T-1,P,D2), D2 < D1.
:- moves(T), disk(D), goal(G), not on(T,G,D).

#show on/3.
#show move/3.
#show moves/1.