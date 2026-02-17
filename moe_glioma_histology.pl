%========================
% moe_glioma_histology.pl
% DeepProbLog / ProbLog program
%========================

% --- Domains (binary outcomes for each expert) ---
outcome(present).
outcome(absent).

% --- Neural experts ---
% Each line defines a neural annotated disjunction (nAD):
% nn(ModelName, [Inputs], OutputVar, Domain) :: Predicate(Inputs, OutputVar).

nn(necrosis_net,        [I], O, [present,absent]) :: necrosis(I,O).
%nn(palisading_net,      [I], O, [present,absent]) :: pseudopalisading(I,O).
nn(mvp_net,             [I], O, [present,absent]) :: microvascular_prolif(I,O).
%nn(vascularity_net,     [I], O, [present,absent]) :: high_vascularity(I,O).

nn(hemorrhage_net,      [I], O, [present,absent]) :: hemorrhage(I,O).
nn(thrombosis_net,      [I], O, [present,absent]) :: thrombosis(I,O).

nn(normal_net,          [I], O, [present,absent]) :: normal_parenchyma(I,O).
%nn(artifact_net,        [I], O, [present,absent]) :: artifact(I,O).

nn(hypercell_net,       [I], O, [present,absent]) :: hypercellularity(I,O).
nn(pleomorphism_net,    [I], O, [present,absent]) :: pleomorphism(I,O).
nn(mitosis_net,         [I], O, [present,absent]) :: mitoses(I,O).

%nn(reactive_net,        [I], O, [present,absent]) :: reactive_gliosis(I,O).
nn(edema_net,           [I], O, [present,absent]) :: edema_or_microcysts(I,O).
%nn(calcification_net,   [I], O, [present,absent]) :: calcification_or_hyalin(I,O).
nn(infiltration_net,    [I], O, [present,absent]) :: infiltration_cues(I,O).

% -------------------------
% Quality gating
% -------------------------
%good_quality(I) :- artifact(I, absent).
%bad_quality(I)  :- artifact(I, present).

% -------------------------
% tumor context on H&E
% -------------------------
tumor_context(I) :-
    hypercellularity(I, present);
    pleomorphism(I, present);
    infiltration_cues(I, present);
    mitoses(I, present).

hypoxia_occlusion_context(I) :-
    thrombosis(I, present).

% -------------------------
% Confounder-aware “trusted” criteria
% (Adjust these to match your data; they are conservative.)
% -------------------------
% -------------------------
% Stronger tumor context for patch-level gating
% (At least two cues; approximates "strong context" without counting)
% -------------------------
strong_tumor_context(I) :-
    ( hypercellularity(I, present), pleomorphism(I, present) );
    ( hypercellularity(I, present), infiltration_cues(I, present) );
    ( hypercellularity(I, present), mitoses(I, present) );
    ( pleomorphism(I, present), infiltration_cues(I, present) );
    ( pleomorphism(I, present), mitoses(I, present) );
    ( infiltration_cues(I, present), mitoses(I, present) ).

% -------------------------
% Necrosis: strict (no hemorrhage) path stays as-is
% -------------------------
trusted_necrosis_strict(I) :-
    necrosis(I, present),
    hemorrhage(I, absent),
    edema_or_microcysts(I, absent),
    normal_parenchyma(I, absent),
    ( strong_tumor_context(I) ; hypoxia_occlusion_context(I) ).

% -------------------------
% Necrosis: hemorrhage-allowed path (patch-level conservative)
% Requires stronger corroboration to avoid RBC-lake / clot confusion.
% -------------------------
trusted_necrosis_hemorrhagic(I) :-
    necrosis(I, present),
    hemorrhage(I, present),
    edema_or_microcysts(I, absent),
    normal_parenchyma(I, absent),
    (
        thrombosis(I, present)
      ; strong_tumor_context(I)
    ).

% -------------------------
% Unified trusted necrosis
% -------------------------
trusted_necrosis(I) :- trusted_necrosis_strict(I).
trusted_necrosis(I) :- trusted_necrosis_hemorrhagic(I).


% MVP should only be trusted when:
% - quality is good
% - MVP is predicted present
% - tissue is predicted not-normal
% - and there is strong tumor context (>=2 cues)

trusted_mvp(I) :-
    microvascular_prolif(I, present),
    normal_parenchyma(I, absent),
    tumor_context(I).


% -------------------------
% WHO-grade-4 histology feature detector (slide-only)
% Grade-4 histologic features are necrosis and (or) microvascular proliferation.
% This program only outputs "grade4 features present" rather than an integrated entity.
% -------------------------
grade4_features_present(I) :- trusted_necrosis(I).
grade4_features_present(I) :- trusted_mvp(I).

% -------------------------
% If quality is OK and neither criterion is present, call "not detected"
% Note: uses stratified negation over derived predicates.
% -------------------------
grade4_features_not_detected(I) :-
    \+trusted_necrosis(I),
    \+trusted_mvp(I).

% -------------------------
% Final prediction with three mutually exclusive outcomes
% -------------------------
prediction(I, grade4_features_present) :- grade4_features_present(I).
prediction(I, grade4_features_not_detected) :- grade4_features_not_detected(I).

% -------------------------
% Explanation predicates (what fired)
% These are useful to query alongside prediction/2
% -------------------------
evidence(I, necrosis) :- trusted_necrosis(I).
evidence(I, microvascular_prolif) :- trusted_mvp(I).

% Supporting, non-decisive evidence (optional)
%support(I, pseudopalisading) :- pseudopalisading(I, present)
support(I, thrombosis) :- thrombosis(I, present).
support(I, hemorrhage) :- hemorrhage(I, present).
support(I, hypercellularity) :- hypercellularity(I, present).
support(I, pleomorphism) :- pleomorphism(I, present).
support(I, mitoses) :- mitoses(I, present).
support(I, infiltration_cues) :- infiltration_cues(I, present).
%support(I, reactive_or_inflam) :- reactive_gliosis(I, present).

% -------------------------
% Queries you typically run
% -------------------------
% query(prediction(I, C)).
% query(evidence(I, E)).
% query(support(I, S)).
