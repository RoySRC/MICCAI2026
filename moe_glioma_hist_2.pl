%========================
% moe_glioma_histology_fixed_cyclefree.pl
% Cycle-free: no mutual negation, no negation loops.
%========================

outcome(present).
outcome(absent).

nn(necrosis_net,     [I], O, [present,absent]) :: necrosis(I,O).
nn(mvp_net,          [I], O, [present,absent]) :: microvascular_prolif(I,O).
nn(normal_net,       [I], O, [present,absent]) :: normal_parenchyma(I,O).

nn(hypercell_net,    [I], O, [present,absent]) :: hypercellularity(I,O).
nn(pleomorphism_net, [I], O, [present,absent]) :: pleomorphism(I,O).
nn(mitosis_net,      [I], O, [present,absent]) :: mitoses(I,O).
nn(infiltration_net, [I], O, [present,absent]) :: infiltration_cues(I,O).

% Context (non-hallmark)
tumor_context(I) :-
    hypercellularity(I, present);
    pleomorphism(I, present);
    infiltration_cues(I, present);
    mitoses(I, present).

non_normal(I) :-
    normal_parenchyma(I, absent).

% Hallmark "trusted" detections
trusted_necrosis(I) :-
    necrosis(I, present).

trusted_mvp(I) :-
    microvascular_prolif(I, present).

% ------------------------
% Grade-4 rule: DISJUNCTION of hallmarks
% Option A (pure): hallmark alone is enough
grade4_features_present(I) :-
    trusted_necrosis(I) ;
    trusted_mvp(I).

% Option B (if you truly want tumor evidence, but avoid hard gating):
% add extra proofs rather than an AND gate. This biases probability upward
% when context is present, without vetoing positives.
grade4_features_present(I) :-
    ( trusted_necrosis(I) ; trusted_mvp(I) ),
    ( tumor_context(I) ; non_normal(I) ).

% Negative label as simple complement (single negation, one-way)
grade4_features_not_detected(I) :-
    \+ grade4_features_present(I).

% Predictions
prediction(I, grade4_features_present) :-
    grade4_features_present(I).

prediction(I, grade4_features_not_detected) :-
    grade4_features_not_detected(I).

% Evidence hooks
evidence(I, necrosis) :-
    trusted_necrosis(I).

evidence(I, microvascular_prolif) :-
    trusted_mvp(I).

support(I, tumor_context) :-
    tumor_context(I).

support(I, non_normal) :-
    non_normal(I).

support(I, hypercellularity) :-
    hypercellularity(I, present).

support(I, pleomorphism) :-
    pleomorphism(I, present).

support(I, mitoses) :-
    mitoses(I, present).

support(I, infiltration_cues) :-
    infiltration_cues(I, present).