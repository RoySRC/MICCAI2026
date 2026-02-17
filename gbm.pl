% gbm.pl

% Neural annotated disjunctions (nADs) for binary concept presence.
% Domain is [0,1] where 1 means present.
nn(mvp_net,      [I], Y, [0,1]) :: mvp(I,Y).
nn(necrosis_net, [I], Y, [0,1]) :: necrosis(I,Y).

% gbm is true if mvp is present OR necrosis is present (or both)
gbm(I) :- ( mvp(I,1) ; necrosis(I,1) ).

% Query declaration is optional; training uses Query objects in Python.