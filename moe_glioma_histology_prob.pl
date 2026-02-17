%========================
% moe_glioma_histology_prob.pl
% DeepProbLog / ProbLog program
%========================

% Discrete outcome domains
state(present).
state(absent).

%------------------------
% Neural experts (each returns P(present) and P(absent))
%------------------------
nn(artifact_net,      [I], S, [present,absent]) :: artifact(I,S).

nn(necrosis_net,      [I], S, [present,absent]) :: necrosis(I,S).
nn(mvp_net,           [I], S, [present,absent]) :: mvp(I,S).
nn(palisading_net,    [I], S, [present,absent]) :: palisading(I,S).

nn(hemorrhage_net,    [I], S, [present,absent]) :: hemorrhage(I,S).
nn(edema_net,         [I], S, [present,absent]) :: edema(I,S).

% Optional supporting experts
nn(thrombosis_net,    [I], S, [present,absent]) :: thrombosis(I,S).
nn(vascularity_net,   [I], S, [present,absent]) :: high_vascularity(I,S).
nn(reactive_net,      [I], S, [present,absent]) :: reactive(I,S).

%------------------------
% Soft “quality” concept
%------------------------
good_quality(I) :- artifact(I, absent).

%------------------------
% Confounder-aware criteria
% Keep these if you want conservative decisions.
% If you want permissive decisions, drop the confounder clauses.
%------------------------
trusted_necrosis(I) :-
    necrosis(I, present),
    hemorrhage(I, absent),
    edema(I, absent).

trusted_mvp(I) :-
    mvp(I, present).

%------------------------
% WHO-grade-4 histology feature event
% (Necrosis and or MVP on acceptable quality)
%------------------------
grade4_event(I) :- good_quality(I), trusted_necrosis(I).
grade4_event(I) :- good_quality(I), trusted_mvp(I).

%------------------------
% Final diagnosis categories (mutually exclusive)
% In ProbLog, these become probabilistic because they depend on probabilistic facts.
%------------------------
diagnosis(I, insufficient_quality) :- artifact(I, present).
diagnosis(I, grade4_features_present) :- grade4_event(I), artifact(I, absent).
diagnosis(I, grade4_features_not_detected) :-
    artifact(I, absent),
    \+ grade4_event(I).

%------------------------
% Explanation atoms (auditable)
%------------------------
evidence(I, necrosis) :- trusted_necrosis(I).
evidence(I, mvp) :- trusted_mvp(I).
support(I, palisading) :- palisading(I, present), good_quality(I).
support(I, thrombosis) :- thrombosis(I, present), good_quality(I).

% Typical queries:
% query(diagnosis(I, insufficient_quality)).
% query(diagnosis(I, grade4_features_present)).
% query(diagnosis(I, grade4_features_not_detected)).
% query(evidence(I, E)).