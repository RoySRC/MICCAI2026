%========================
% moe_glioma_histology_rewrite.pl
% DeepProbLog / ProbLog program
%
% Design goal:
% - Encode WHO-grade-4 histologic hallmarks as NECROSIS and/or MVP only.
% - Treat hypercellularity/pleomorphism/mitoses/infiltration as CONTEXT, not as a grade-4 hallmark.
% - Avoid “collapse-to-absent” incentives by not using confounders only as negated literals.
%========================

outcome(present).
outcome(absent).

nn(necrosis_net,        [I], O, [present,absent]) :: necrosis(I,O).
nn(mvp_net,             [I], O, [present,absent]) :: microvascular_prolif(I,O).

%nn(hemorrhage_net,      [I], O, [present,absent]) :: hemorrhage(I,O).
%nn(thrombosis_net,      [I], O, [present,absent]) :: thrombosis(I,O).

nn(normal_net,          [I], O, [present,absent]) :: normal_parenchyma(I,O).

nn(hypercell_net,       [I], O, [present,absent]) :: hypercellularity(I,O).
nn(pleomorphism_net,    [I], O, [present,absent]) :: pleomorphism(I,O).
nn(mitosis_net,         [I], O, [present,absent]) :: mitoses(I,O).

% Edema is commented out because this is a MR finding.
%nn(edema_net,           [I], O, [present,absent]) :: edema_or_microcysts(I,O).
nn(infiltration_net,    [I], O, [present,absent]) :: infiltration_cues(I,O).

% Context (non-hallmark)
tumor_context(I) :-
    hypercellularity(I, present);
    pleomorphism(I, present);
    infiltration_cues(I, present);
    mitoses(I, present).

strong_tumor_context_2(I) :-
    ( hypercellularity(I, present), pleomorphism(I, present) );
    ( hypercellularity(I, present), infiltration_cues(I, present) );
    ( hypercellularity(I, present), mitoses(I, present) );
    ( pleomorphism(I, present), infiltration_cues(I, present) );
    ( pleomorphism(I, present), mitoses(I, present) );
    ( infiltration_cues(I, present), mitoses(I, present) ).

strong_tumor_context_3(I) :-
    ( hypercellularity(I, present), pleomorphism(I, present), infiltration_cues(I, present) );
    ( hypercellularity(I, present), pleomorphism(I, present), mitoses(I, present) );
    ( hypercellularity(I, present), infiltration_cues(I, present), mitoses(I, present) );
    ( pleomorphism(I, present), infiltration_cues(I, present), mitoses(I, present) ).

non_normal(I) :- normal_parenchyma(I, absent).

% Hallmark candidates require non-normal tissue
necrosis_candidate(I) :- necrosis(I, present), non_normal(I).
mvp_candidate(I)      :- microvascular_prolif(I, present), non_normal(I).

% Trusted necrosis (confounder-aware)
necrosis_supported(I) :-
    necrosis_candidate(I).


trusted_necrosis(I) :-
    necrosis_supported(I).

% Trusted MVP (lighter gating; still needs tumor context)
mvp_supported(I) :-
    mvp_candidate(I).

trusted_mvp(I) :-
    mvp_supported(I).

% Grade-4 histologic features: necrosis and/or MVP
grade4_features_present(I) :- 
    trusted_necrosis(I), trusted_mvp(I),
    (tumor_context(I); strong_tumor_context_2(I) ; strong_tumor_context_3(I) ).

grade4_features_not_detected(I) :-
    \+necrosis_candidate(I),
    \+mvp_candidate(I),
    \+non_normal(I),
    pleomorphism(I, absent),
    infiltration_cues(I, absent),
    \+tumor_context(I),
    \+strong_tumor_context_2(I),
    \+strong_tumor_context_3(I),
    \+trusted_necrosis(I), 
    \+trusted_mvp(I).

prediction(I, grade4_features_present) :- grade4_features_present(I).
prediction(I, grade4_features_not_detected) :- grade4_features_not_detected(I).

evidence(I, necrosis) :- trusted_necrosis(I).
evidence(I, microvascular_prolif) :- trusted_mvp(I).

support(I, tumor_context) :- tumor_context(I).
support(I, strong_tumor_context_2) :- strong_tumor_context_2(I).
support(I, strong_tumor_context_3) :- strong_tumor_context_3(I).
%support(I, thrombosis) :- thrombosis(I, present).
support(I, hemorrhage) :- hemorrhage(I, present).
%support(I, edema_or_microcysts) :- edema_or_microcysts(I, present).
support(I, hypercellularity) :- hypercellularity(I, present).
support(I, pleomorphism) :- pleomorphism(I, present).
support(I, mitoses) :- mitoses(I, present).
support(I, infiltration_cues) :- infiltration_cues(I, present).

