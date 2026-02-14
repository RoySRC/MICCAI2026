"""
concept_prompt_pack.py

Generates and maintains a strict prompt pack for HSI histology concept embeddings:
- Expert: Microvascular Proliferation (MVP)
- Expert: Necrosis (non-palisading)

Outputs:
- positives (concept-present)
- hard_negatives (competing morphology descriptions without naming the competing hallmark)
- strict linter blacklist (global + per-expert + hard-negative integrity)

Optionally expands prompt banks using GPT-5.2 (thinking) via the Responses API.
"""

from __future__ import annotations

import sys
import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Optional dependency: OpenAI Python SDK
# pip install --upgrade openai
# Responses API reference
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# Optional dependency: Hugging Face transformers (for CLIP token budgeting)
# pip install --upgrade transformers
try:
    from transformers import AutoConfig, AutoTokenizer
except Exception:
    AutoConfig = None  # type: ignore
    AutoTokenizer = None  # type: ignore


# =============================================================================
# 1) Modality phrase: pick one and stick to it across ALL prompts
# =============================================================================
MODALITY_PHRASE = ""

# CLIP text token budget.
# Most CLIP text encoders (including PLIP) use max_position_embeddings = 77.
# We enforce this at PROMPT GENERATION time so embedding does not require truncation.
CLIP_TEXT_MODEL_ID = os.environ.get("CLIP_TEXT_MODEL_ID", "vinid/plip")
CLIP_MAX_TOKENS = int(os.environ.get("CLIP_MAX_TOKENS", "77"))


# Alternative allowed phrases (do not mix; pick one constant above):
# "HSI microscopy patch of brain tissue, false-color composite"
# "Hyperspectral pathology patch (HSI), band-composite rendering"


# =============================================================================
# 2) Seed prompt packs (exactly reflecting your provided prompts)
# =============================================================================

@dataclass(frozen=True)
class ExpertPack:
    name: str
    positives: List[str]
    positives_short: List[str]
    hard_negatives: List[str]
    hard_negatives_short: List[str]  # optional, can be empty


def _prefix_with_modality(prompt: str) -> str:
    """
    Ensures the fixed modality phrase appears explicitly.
    We do not enforce exact prefix placement, but enforce presence.
    """
    return prompt
    if "HSI" in prompt or "Hyperspectral" in prompt:
        return prompt
    return f"{MODALITY_PHRASE}, {prompt}"


ARTIFACT_PACK = ExpertPack(
    name="artifact",
    positives=[
        "out-of-focus blur with smeared texture and loss of crisp cellular edges.",
        "directional motion-like smearing across the field with reduced fine detail.",
        "a saturated bright region that obscures underlying tissue structure.",
        "repetitive horizontal banding across the image that does not follow tissue structure.",
        "an abrupt boundary with duplicated or misaligned structures along a straight seam-like edge.",
        "peripheral darkening with reduced contrast toward the edges while the center remains clearer.",
        "salt-and-pepper speckling inconsistent with biologic microtexture.",
        "a sharp non-biologic line or spot consistent with foreign particulate contamination.",
    ],
    positives_short=[
        "out-of-focus blur with loss of fine detail",
        "directional smearing with reduced sharpness",
        "saturated bright region obscuring tissue",
        "repetitive banding not aligned to tissue",
        "abrupt seam with duplicated structures",
        "peripheral darkening with reduced contrast",
        "salt-and-pepper speckling pattern",
        "foreign particulate contamination line",
    ],
    hard_negatives=[
        "uniformly sharp cellular detail with consistent contrast across the field.",
        "continuous tissue texture without abrupt boundaries or duplicated structures.",
        "compact rounded clusters of many small circular lumina tightly grouped together.",
        "a large pale region with minimal intact cells and scattered nuclear fragments, with coherent tissue context around it.",
        "dense cellular regions with crisp nuclei and intact microarchitecture throughout the view.",
        "multiple closely packed ring-like luminal profiles within a small region, with stable sharpness across the scene.",
    ],
    hard_negatives_short=[],
)

MVP_PACK = ExpertPack(
    name="microvascular proliferation",
    positives=[
        "tufted clusters of small vessels forming rounded vascular aggregates.",
        "glomeruloid vascular tufts composed of crowded capillary loops.",
        "multiple closely packed vascular lumina within a compact vascular cluster.",
        "abnormally dense microvasculature with complex, tufted vessel profiles.",
        "proliferative capillary tufts with piled-up endothelial nuclei around small lumina.",
        "tangled, multilayered-appearing microvessels forming a compact vascular nidus.",
        "irregular, crowded capillary networks with rounded tuft-like vascular formations.",
        "microvascular hyperplasia with clustered capillary loops and thickened vascular profiles.",
    ],
    positives_short=[
        "glomeruloid vascular tufts",
        "tufted proliferative capillary loops",
        "crowded microvascular clusters with multiple lumina",
    ],
    hard_negatives=[
        "a large acellular pale-appearing area with minimal tissue texture and very sparse nuclei.",
        "amorphous granular debris with loss of recognizable cellular architecture.",
        "broad regions with markedly reduced nuclear density and little structural detail.",
        "cell dropout with scattered nuclear fragments and diffuse background debris.",
        "featureless low-texture zones lacking intact cellular structures.",
        "geographic acellular regions with only scattered karyorrhectic debris.",
    ],
    hard_negatives_short=[],
)

NECROSIS_PACK = ExpertPack(
    name="necrosis",
    positives=[
        "geographic tissue breakdown with loss of intact nuclei and amorphous debris.",
        "acellular granular material with fragmented nuclear debris and reduced tissue structure.",
        "coagulative-type tissue breakdown with ghost-like outlines and sparse viable cells.",
        "broad acellular zones with karyorrhectic debris and architectural collapse.",
        "cellular dissolution with scattered nuclear fragments in a low-texture background.",
        "large areas with absent viable cellular detail and abundant granular debris.",
        "acellular fields with minimal nuclei and disrupted microarchitecture.",
        "tissue breakdown characterized by loss of nuclei and amorphous proteinaceous debris.",
    ],
    positives_short=[
        "geographic tissue breakdown with acellular debris",
        "karyorrhectic debris with loss of architecture",
        "acellular low-texture zone with sparse nuclei",
    ],
    hard_negatives=[
        "compact rounded clusters of many small circular lumina tightly grouped together.",
        "dense networks of tiny vessel-like loops forming a tight, tuft-like cluster.",
        "multiple closely packed ring-like luminal profiles within a small region.",
        "crowded tubular profiles with repeated small lumina and thickened outlines.",
        "a focal area with many closely spaced lumen-like structures forming a compact aggregate.",
        "tangled small luminal loops grouped into a rounded nidus-like cluster.",
    ],
    hard_negatives_short=[],
)

HEMORRHAGE_PACK = ExpertPack(
    name="hemorrhage",
    positives=[
        "extravascular pools of intact red blood cells within the tissue space.",
        "diffuse interstitial red blood cells tracking through the parenchyma rather than confined to vessel lumens.",
        "a focal blood lake with dense packed erythrocytes outside vascular boundaries.",
        "scattered microfoci of red blood cell extravasation around disrupted microarchitecture.",
        "fresh-appearing hemorrhage with abundant extravascular erythrocytes and minimal surrounding cellular detail change.",
        "patchy hemorrhage with irregular collections of red blood cells in the interstitium.",
        "hemorrhagic foci with extravascular erythrocytes and focal clot-like density.",
        "prior hemorrhage suggested by pigment-laden macrophage-like cells adjacent to residual extravascular red blood cells.",
    ],
    positives_short=[
        "extravascular red blood cell pools",
        "interstitial erythrocyte extravasation",
        "focal blood lake outside vessel lumens",
    ],
    hard_negatives=[
        "a broad pale low-texture zone with markedly reduced recognizable cellular architecture.",
        "amorphous granular background with sparse intact nuclei and loss of organized tissue pattern.",
        "a geographic region of cell dropout with scattered nuclear fragments and minimal structured detail.",
        "compact rounded clusters of many small circular lumina tightly grouped in a focal aggregate.",
        "dense networks of tiny loop-like luminal profiles forming a tight rounded cluster.",
        "repeated small ring-like lumen profiles with thickened outlines concentrated in a small nidus-like region.",
    ],
    hard_negatives_short=[],
)

THROMBOSIS_PACK = ExpertPack(
    name="thrombosis",
    positives=[
        "an intravascular lumen partially occluded by an adherent clot-like mass attached to the vessel wall.",
        "a vessel lumen filled by a fibrin-platelet–rich appearing plug with trapped cellular elements.",
        "a mural thrombus pattern with layered intravascular material lining one side of the vessel lumen.",
        "a lumen-filling intravascular coagulum with a sharp interface between the vessel wall and the occluding material.",
        "laminated intravascular material with alternating pale and darker bands within an occluded vessel lumen.",
        "a focal intravascular occlusion where the lumen is bridged by dense strand-like material forming a plug.",
        "a compact intravascular blockage with residual narrow flow channel at the periphery of the occluded lumen.",
        "an intravascular clot occupying the lumen with a cohesive, matrix-like texture distinct from surrounding tissue.",
    ],
    positives_short=[
        "intravascular lumen-filling clot attached to vessel wall",
        "occlusive fibrin-platelet plug within a vessel lumen",
        "laminated intravascular clot with banded appearance",
    ],
    hard_negatives=[
        "a broad pale low-texture zone with markedly reduced recognizable cellular architecture.",
        "amorphous granular background with sparse intact nuclei and loss of organized tissue pattern.",
        "diffuse interstitial red blood cells dispersed through tissue spaces rather than confined within vessel boundaries.",
        "a focal pool of densely packed red blood cells occupying the tissue space outside vascular outlines.",
        "compact rounded clusters of many small circular lumina tightly grouped in a focal aggregate.",
        "dense networks of tiny loop-like luminal profiles forming a tight rounded cluster.",
    ],
    hard_negatives_short=[],
)

HIGH_VASCULARITY_PACK = ExpertPack(
    name="high vascularity",
    positives=[
        "diffusely increased vessel density with many separate vascular profiles distributed across the field.",
        "numerous thin-walled vascular cross-sections with open lumina, evenly spaced rather than forming a focal cluster.",
        "a broad increase in visible vessel profiles throughout the view without a compact rounded vascular nodule.",
        "many patent luminal rings scattered across the tissue with no tight knot-like grouping.",
        "prominent background vascularity with repeated isolated vessel cross-sections across the entire view.",
        "widespread vascular channels with clear lumina and regular spacing, not confined to a single focal aggregate.",
        "increased vascular visibility across the tissue microarchitecture with vessels appearing as separate profiles rather than a compact cluster.",
        "increased vessel density with many distinct vessel lumina diffusely distributed and not forming a rounded cluster.",
    ],
    positives_short=[
        "diffuse increased vessel density with separate vascular profiles",
        "many isolated luminal rings distributed across the field",
        "widespread vascular channels without a focal compact cluster",
    ],
    hard_negatives=[
        "a focal compact round region packed with many tiny circular lumina tightly grouped together.",
        "a small localized knot of closely packed ring-like luminal profiles within one confined area.",
        "a dense localized aggregate of repeated small lumen-like circles forming a rounded nodule.",
        "a tight focal cluster of many closely spaced lumina with minimal separation between luminal rings.",
        "a localized rounded tangle of tiny luminal loops concentrated into a single compact focus.",
        "a focal region dominated by densely grouped luminal rings rather than diffuse vessel distribution.",
    ],
    hard_negatives_short=[],
)

NORMAL_PARENCHYMA_PACK = ExpertPack(
    name="normal parenchyma",
    positives=[
        "preserved parenchymal texture with a fine fibrillary background and evenly distributed small nuclei.",
        "intact microarchitecture with scattered neuronal cell bodies and supporting glial nuclei in a dense neuropil-like background.",
        "uniform nuclear density and preserved tissue structure without focal low-texture zones.",
        "normal-appearing parenchyma with delicate capillary profiles that are spaced apart and not clustered.",
        "preserved neuropil with fine background texture, evenly spaced nuclei, and intact microarchitecture, without large contiguous regions of tissue loss.",
        "preserved parenchymal organization without architectural disruption.",
        "intact parenchymal fields with fine texture and consistent cellular spacing.",
        "evenly scattered small nuclei in preserved parenchyma without focal abnormal aggregates.",
    ],
    positives_short=[
        "preserved neuropil-like background with evenly scattered small nuclei",
        "intact parenchymal microarchitecture with uniform cellular spacing",
        "delicate non-clustered capillary profiles in preserved parenchyma",
    ],
    hard_negatives=[
        "a large pale-appearing region with minimal tissue texture and markedly reduced nuclear detail.",
        "amorphous granular material with loss of recognizable microarchitecture and scattered nuclear fragments.",
        "broad low-texture zones with sparse nuclei and diffuse background particulate material.",
        "tissue dropout with disrupted structure and scattered nuclear fragments in the background.",
        "compact rounded clusters of many small circular lumina tightly grouped within a small region.",
        "dense networks of tiny ring-like luminal profiles forming a compact aggregate.",
        "multiple closely packed lumen-like structures repeated across a focal area.",
        "crowded tubular outlines with repeated small lumina concentrated in one focus.",
    ],
    hard_negatives_short=[],
)

HYPERCELLULARITY_PACK = ExpertPack(
    name="hypercellularity",
    positives=[
        "diffuse cellular crowding with markedly increased nuclear density and minimal intercellular spacing.",
        "sheets of closely packed nuclei with reduced background space and compressed tissue texture.",
        "a uniformly cell-dense field dominated by near-contiguous nuclei and little visible extracellular space.",
        "a hyperdense nuclear pattern with frequent nuclear overlap and crowded cellular architecture.",
        "high nuclear density with broadly increased cell density across the field and minimal empty space between nuclei.",
        "packed small-to-medium nuclei with frequent overlap and a tightly crowded microarchitecture.",
        "high cellularity with tightly apposed nuclei and reduced visible intercellular background.",
        "cellular sheets with minimal spacing and a crowded nuclear landscape throughout the view.",
    ],
    positives_short=[
        "markedly increased nuclear density with minimal spacing",
        "diffuse cellular crowding with overlapping nuclei",
        "cell-dense field with near-contiguous nuclei",
    ],
    hard_negatives=[
        # Competing morphology without naming competing hallmarks:
        # (A) necrotic-like morphology (do not name it)
        "a broad low-texture zone with very few intact nuclei and diffuse granular background material.",
        "featureless pale-appearing regions with loss of recognizable cellular architecture and scattered nuclear fragments.",
        # (B) MVP-like morphology (do not name it)
        "a compact rounded aggregate of many small ring-like luminal profiles tightly grouped within a small region.",
        "dense networks of tiny looped tubular profiles forming a tight clustered pattern with repeated small lumina.",
        # (C) hypocellular / low-density field
        "widely spaced nuclei with abundant open background space and a loose, low-density cellular pattern.",
        "sparse scattered nuclei separated by large empty-appearing areas, without diffuse nuclear crowding.",
    ],
    hard_negatives_short=[],
)

PLEOMORPHISM_PACK = ExpertPack(
    name="pleomorphism",
    positives=[
        "marked variation in nuclear size and shape with enlarged irregular hyperchromatic nuclei interspersed with smaller nuclei.",
        "scattered very large atypical nuclei with irregular contours and striking heterogeneity across the field.",
        "anisokaryosis and anisocytosis with uneven nuclear contours and variable chromatin texture.",
        "bizarre enlarged nuclei with lobulated outlines and mixed nuclear sizes within the same region.",
        "a heterogeneous cell population with extreme nuclear size variability and irregular nuclear shapes.",
        "occasional multinucleated giant cell forms and markedly atypical nuclear morphology.",
        "pronounced variability in nuclear staining intensity with irregular nuclear contours and coarse chromatin patterns.",
        "diffuse nuclear atypia with mixed small and very large nuclei and distorted nuclear outlines.",
    ],
    positives_short=[
        "marked variation in nuclear size and shape",
        "bizarre enlarged hyperchromatic nuclei with irregular contours",
        "scattered multinucleated giant cell forms with extreme nuclear atypia",
    ],
    hard_negatives=[
        "broad low-texture regions with markedly reduced nuclear density and minimal recognizable cellular detail.",
        "featureless pale-appearing zones with scattered nuclear fragments and diffuse granular background material.",
        "compact rounded clusters of many small circular lumina tightly grouped together in a focal aggregate.",
        "repeated small ring-like luminal profiles densely packed within a small region with thickened outlines.",
        "a uniform field of evenly spaced small round nuclei with minimal variation in size or shape across the view.",
        "monomorphic nuclei with consistent size, shape, and chromatin appearance throughout the region.",
    ],
    hard_negatives_short=[],
)

MITOSES_PACK = ExpertPack(
    name="mitoses",
    positives=[
        "multiple mitotic figures with condensed chromatin and clear metaphase or anaphase configurations.",
        "frequent mitotic figures with dark condensed nuclear material and division-phase morphology.",
        "several mitotic figures with hyperchromatic condensed chromosomes and visible mitotic spindling patterns in nearby cells.",
        "brisk mitotic activity with multiple cells in mitosis, including metaphase plates and anaphase separation.",
        "numerous mitotic figures scattered across the field, with compact chromatin clumps and division-phase nuclear shapes.",
        "repeated mitotic figures per field, including atypical mitotic forms with irregular condensed chromatin groupings.",
        "many mitotic figures among densely packed nuclei, with distinct condensed chromatin bodies consistent with active cell division.",
        "frequent mitoses with sharply condensed chromatin and characteristic rounded division-phase nuclei.",
    ],
    positives_short=[
        "frequent mitotic figures with condensed chromatin",
        "multiple cells in metaphase or anaphase",
        "brisk mitotic activity per field",
    ],
    hard_negatives=[
        "a large low-texture region with markedly reduced nuclear detail and sparse intact cellular structures.",
        "amorphous granular background material with loss of recognizable cellular architecture and only scattered nuclear fragments.",
        "broad areas of cell dropout with minimal tissue texture and few intact nuclei across the region.",
        "compact rounded clusters of many small circular lumina tightly grouped together within a focal aggregate.",
        "dense networks of tiny vessel-like loops forming a tight cluster with repeated small lumen-like profiles.",
        "multiple closely packed ring-like luminal profiles within a compact region, forming a rounded clustered pattern.",
    ],
    hard_negatives_short=[],
)

INFILTRATION_PACK = ExpertPack(
    name="infiltration cues",
    positives=[
        "scattered atypical cells permeating the neuropil between intact background structures, extending beyond the main cellular cluster.",
        "diffuse intermingling of neoplastic-appearing cells with preserved parenchymal texture, without a sharp boundary.",
        "tumor cells tracking along a fibrillary background in thin linear streams, consistent with spread along tissue architecture.",
        "perineuronal satellitosis, with small tumor cells clustering around neurons within otherwise preserved parenchyma.",
        "perivascular cuffing by dispersed tumor cells around small vessels within adjacent tissue.",
        "an infiltrative edge with tapering cellular density and isolated cells dispersed into surrounding tissue.",
        "isolated tumor cells infiltrating along fiber-like tracts with irregular spacing and preserved background texture.",
        "a diffuse cellular spread pattern with background elements still visible between tumor cells, lacking a pushing border.",
    ],
    positives_short=[
        "single-cell infiltration through neuropil",
        "perineuronal satellitosis in adjacent parenchyma",
        "perivascular cuffing by dispersed tumor cells",
    ],
    hard_negatives=[
        "a sharply demarcated compact cellular mass with a clean boundary and no dispersed cells beyond the edge.",
        "cohesive sheets of densely packed cells with uniform texture across the field, without intermingling background structures.",
        "a well-circumscribed nodule with abrupt transition to normal-appearing tissue outside the border.",
        "a compact cellular aggregate with smooth margins and absence of satellite cells around neurons.",
        "a localized cellular cluster centered in the field with surrounding tissue largely spared and no perivascular cellular cuffs.",
        "a pushing-border growth pattern with a continuous cellular front and minimal isolated cells in adjacent tissue.",
    ],
    hard_negatives_short=[],
)

CALCIFICATION_PACK = ExpertPack(
    name="calcification",
    positives=[
        "punctate coarse granular mineral-like deposits embedded within the tissue.",
        "dense speckled deposits consistent with mineral accumulation in small focal clusters.",
        "discrete calcified-appearing granules forming compact, irregular deposits.",
        "laminated mineralized-appearing bodies with concentric layering in a focal region.",
        "coarse chalky-appearing deposits with sharp granular texture compared to surrounding parenchyma.",
        "scattered mineralized deposits with a gritty granular appearance and minimal associated cellular detail.",
        "focal dystrophic-appearing mineral deposits within disrupted tissue matrix.",
        "clustered calcified-appearing granules forming a small dense deposit amid surrounding tissue texture.",
    ],
    positives_short=[
        "punctate calcified foci with coarse granular texture",
        "coarse granular mineral deposits in focal clusters",
        "laminated mineralized bodies with concentric layering",
    ],
    hard_negatives=[
        # Competing morphology: vessel-cluster patterns (MVP-like) without naming the hallmark
        "compact rounded clusters of many small circular lumina tightly grouped together.",
        "dense networks of tiny loop-like luminal profiles forming a tight rounded cluster.",
        "multiple closely packed ring-like luminal structures within a small focal region.",
        # Competing morphology: tissue breakdown patterns (necrotic-like) without using the hallmark name
        "a broad low-texture region with markedly reduced nuclear detail and sparse recognizable architecture.",
        "amorphous granular background material with scattered nuclear fragments and loss of intact microarchitecture.",
        "geographic tissue dropout with minimal intact cellular structures and diffuse background particulate material.",
    ],
    hard_negatives_short=[],
)

MICROCYSTS_PACK = ExpertPack(
    name="microcysts",
    positives=[
        "numerous small round clear spaces creating a bubbly microcystic pattern within the tissue.",
        "many tiny fluid-like vacuole-shaped spaces separated by thin strands of cellular tissue.",
        "a swiss-cheese-like field of small circular empty-appearing spaces with scattered nuclei between the spaces.",
        "microcystic change with multiple small, sharply marginated clear spaces distributed across the view.",
        "clustered microcystic cavities with delicate septa and intermittent cellular islands between clear spaces.",
        "widespread small vacuolated spaces giving a foamy, microcystic tissue texture without a dominant solid sheet.",
        "many discrete round-to-oval clear spaces with a fine reticular background and nuclei lining the intervening tissue strands.",
        "diffuse microcystic spaces of variable size with preserved intervening cellular microarchitecture.",
    ],
    positives_short=[
        "microcystic change with bubbly clear spaces",
        "swiss-cheese pattern of small clear round spaces",
        "numerous tiny vacuole-like spaces across the field",
    ],
    hard_negatives=[
        # Necrosis-like competing morphology, without using the word "necrosis"/related tokens
        "a broad pale low-texture region with loss of recognizable cellular detail and only sparse residual nuclei.",
        "amorphous granular background material with scattered small nuclear fragments and blurred tissue structure.",
        "geographic areas of markedly reduced nuclear detail with diffuse fine particulate material in the background.",
        # MVP-like competing morphology, without using vascular/MVP terms
        "a compact rounded cluster of many small circular ring-like profiles tightly packed into a dense aggregate.",
        "repeated small looped ring-like profiles forming a tight tuft-like cluster within a focal region.",
        "multiple closely packed circular outlines grouped together into a rounded nidus-like aggregate.",
    ],
    hard_negatives_short=[],
)

EDEMA_PACK = ExpertPack(
    name="edema",
    positives=[
        "diffuse pale-appearing interstitial clearing with widened extracellular spaces separating tissue elements.",
        "a spongy low-texture background with clear fluid-like separation between cellular and stromal structures.",
        "perivascular clearing with widened spaces around vessel profiles and reduced local tissue texture.",
        "swollen, loosened-appearing neuropil with increased spacing between nuclei and disrupted compactness of the background matrix.",
        "patchy pale regions with expanded interstitial spaces and vacuolated-appearing separation of tissue elements, with scattered intact nuclei and preserved microarchitecture.",
        "diffuse tissue pallor and interstitial separation producing a washed-out, low-contrast microarchitecture.",
        "broad, faintly textured regions where tissue strands appear separated by clear spaces, consistent with interstitial fluid accumulation.",
        "expanded clear spaces within the tissue background with preserved cellular outlines but reduced overall density and cohesion.",
    ],
    positives_short=[
        "diffuse interstitial clearing with widened extracellular spaces",
        "perivascular clearing with widened perivascular spaces",
        "spongy low-texture background with tissue separation",
    ],
    hard_negatives=[
        # Competing morphology: necrosis-like (do NOT name it)
        "a large featureless pale area with minimal recognizable structure and scattered nuclear fragments in a granular background.",
        "amorphous granular material with loss of intact cellular outlines and a collapsed, disorganized microarchitecture.",
        "broad regions with markedly reduced intact nuclei and diffuse background debris obscuring normal tissue texture.",
        # Competing morphology: MVP-like vascular clustering (do NOT name it)
        "compact rounded clusters of many small circular lumina tightly grouped together within a focal region.",
        "dense networks of tiny ring-like lumen profiles forming a tight rounded cluster with repeated small lumina.",
        "tangled small luminal loops and crowded tubular profiles grouped into a compact nidus-like aggregate.",
    ],
    hard_negatives_short=[],
)

REACTIVE_GLIOSIS_PACK = ExpertPack(
    name="reactive gliosis",
    positives=[
        "reactive astrocytosis with enlarged stellate astrocytes and a fibrillary background of glial processes.",
        "hypertrophic astrocytes with abundant eosinophilic cytoplasm and mildly enlarged, relatively bland nuclei within a dense fibrillary matrix.",
        "scattered plump reactive astrocytes with prominent processes in otherwise preserved tissue architecture.",
        "diffuse fibrillary gliosis with increased fiber-like texture and evenly distributed reactive astrocyte nuclei.",
        "gemistocyte-like reactive astrocytes with glassy eosinophilic cytoplasm and eccentric nuclei in a gliotic background.",
        "reactive glial scarring with thickened astrocytic processes forming a coarse fibrillary network.",
        "mild-to-moderate increase in glial nuclei density composed of hypertrophic astrocytes with elongated processes and relatively preserved microarchitecture.",
        "reactive gliosis characterized by enlarged astrocytic cell bodies, increased fibrillary background, and minimal variation in nuclear size and shape.",
    ],
    positives_short=[
        "reactive astrocytosis with hypertrophic stellate astrocytes in a fibrillary background",
        "plump reactive astrocytes with abundant eosinophilic cytoplasm and prominent processes",
        "diffuse fibrillary gliosis with increased glial fiber texture and preserved architecture",
    ],
    hard_negatives=[
        # Competing morphology (necrotic-like) without naming the hallmark
        "a large pale low-texture zone with marked loss of intact nuclei and abundant granular background material.",
        "geographic areas with minimal cellular structure and scattered nuclear fragments in an amorphous field.",
        # Competing morphology (vascular-tuft-like) without naming the hallmark
        "compact rounded aggregates of many small ring-like luminal profiles tightly clustered within a focal region.",
        "dense networks of tiny tubular loops forming a tight cluster with repeated small lumina and thickened outlines.",
        # Competing morphology (tumor-like hypercellularity/atypia) without diagnosis words
        "sheets of densely packed hyperchromatic nuclei with marked variation in nuclear size and shape and frequent mitotic figures.",
        "diffuse high nuclear density with disorganized architecture and numerous mitotic figures in a crowded cellular field.",
    ],
    hard_negatives_short=[],
)


EXPERTS: Dict[str, ExpertPack] = {
    MVP_PACK.name: MVP_PACK,
    NECROSIS_PACK.name: NECROSIS_PACK,
    HEMORRHAGE_PACK.name: HEMORRHAGE_PACK,
    THROMBOSIS_PACK.name: THROMBOSIS_PACK,
    HIGH_VASCULARITY_PACK.name: HIGH_VASCULARITY_PACK,
    NORMAL_PARENCHYMA_PACK.name: NORMAL_PARENCHYMA_PACK,
    HYPERCELLULARITY_PACK.name: HYPERCELLULARITY_PACK,
    PLEOMORPHISM_PACK.name: PLEOMORPHISM_PACK,
    MITOSES_PACK.name: MITOSES_PACK,
    INFILTRATION_PACK.name: INFILTRATION_PACK,
    ARTIFACT_PACK.name: ARTIFACT_PACK,
    CALCIFICATION_PACK.name: CALCIFICATION_PACK,
    MICROCYSTS_PACK.name: MICROCYSTS_PACK,
    EDEMA_PACK.name: EDEMA_PACK,
    REACTIVE_GLIOSIS_PACK.name: REACTIVE_GLIOSIS_PACK,
}


# =============================================================================
# 3) Strict prompt-linter blacklist (global + per-expert + hard-negative integrity)
# =============================================================================

GLOBAL_BLACKLIST = [
    # Diagnosis/labels
    "glioblastoma", "gbm", "astrocytoma", "oligodendroglioma",
    # Grading
    "who", "grade", "high-grade", "low-grade", "grade iv", "malignant", "grade-4", "grade 4", 
    # Molecular terms
    "idh", "tert", "egfr", "chromosome", "molecular", "mutation", "amplification",
    # Prognostic / diagnostic language
    "prognosis", "survival", "aggressive", "diagnostic", "pathognomonic",
    # Optional ban (recommended in your spec)
    "hypoxia",
]

# Expert-specific blacklists (prevent cross-concept leakage)

ARTIFACT_BLACKLIST = [
    # Necrosis-pattern terms
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "debris", "architectural collapse", "tissue breakdown", "geographic",
    "low nuclei density", "sparse nuclei", "featureless",
    # MVP-pattern terms
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "crowded vessels", "multiple lumina",
]

# For MVP prompt generation: ban necrosis-pattern terms
MVP_BLACKLIST = [
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    # optional strict:
    "debris",
    "architectural collapse", "tissue breakdown", "geographic",
    "low nuclei density", "sparse nuclei", "featureless",
]

# For necrosis prompt generation: ban vascular-proliferation terms
NECROSIS_BLACKLIST = [
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    # optional strict:
    "multiple lumina", "crowded vessels",
]

HEMORRHAGE_BLACKLIST = [
    # Necrosis-pattern vocabulary (ban)
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "architectural collapse", "tissue breakdown", "geographic",
    "low nuclei density", "sparse nuclei", "featureless",
    "debris",  # recommended stricter separation

    # MVP-pattern vocabulary (ban)
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multiple lumina", "crowded vessels",  # optional stricter (enabled here)
]

THROMBOSIS_BLACKLIST = [
    # Necrosis-pattern vocabulary (ban)
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "architectural collapse", "tissue breakdown", "geographic",
    "low nuclei density", "sparse nuclei", "featureless",
    "debris",  # recommended stricter separation

    # MVP-pattern vocabulary (ban)
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multiple lumina", "crowded vessels",  # optional stricter (enabled here)

    # Hemorrhage-pattern vocabulary (ban)
    "hemorrhage", "hemorrhagic", "bleeding",
    "extravascular", "extravasation", "extravasated",
    "blood lake", "interstitial blood", "outside vessel",
    "hemosiderin", "siderophage", "hematoidin",
]

HIGH_VASCULARITY_BLACKLIST = [
    # MVP-style terms
    "microvascular", "microvascular proliferation", "vascular proliferation",
    "glomeruloid",
    "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multilayered", "piled-up",
    "nidus",
    # necrosis-style terms
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost",
    "karyorrhexis", "karyorrhectic",
    "debris",
    "architectural collapse", "tissue breakdown", "geographic",
    "featureless", "sparse nuclei", "low nuclei density",
]

NORMAL_PARENCHYMA_BLACKLIST = [
    # necrosis-pattern vocabulary
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "architectural collapse", "tissue breakdown", "geographic",
    "featureless", "low nuclei density", "sparse nuclei",
    # optional stricter:
    "acellular", "debris", "amorphous", "granular",
    # MVP-pattern vocabulary
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    # optional stricter:
    "multiple lumina", "crowded vessels", "thickened vascular profiles", "vascular nidus",
]

HYPERCELLULARITY_BLACKLIST = [
    # Ban necrosis hallmark naming + necrosis-pattern vocabulary in *hypercellularity positives*
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "debris", "architectural collapse", "tissue breakdown", "geographic",
    "featureless", "low-texture", "sparse nuclei",
    # Ban MVP hallmark naming + MVP-pattern vocabulary in *hypercellularity positives*
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multiple lumina", "crowded vessels",
]

PLEOMORPHISM_BLACKLIST = [
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
]

MITOSES_BLACKLIST = [
    # Competing hallmark names and hallmark-specific vocabulary to prevent cross-concept naming
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "microvascular", "microvascular proliferation", "vascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "mvp",
]

INFILTRATION_BLACKLIST = [
    # necrosis-pattern terms
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "debris", "architectural collapse", "tissue breakdown", "geographic",
    "low nuclei density", "sparse nuclei", "featureless",
    # vascular-proliferation terms
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multiple lumina", "crowded vessels",
]

CALCIFICATION_BLACKLIST = [
    # Ban necrosis hallmark naming and necrosis-pattern vocabulary in calcification POSITIVES
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "architectural collapse", "tissue breakdown", "geographic",
    "low nuclei density", "sparse nuclei", "featureless",

    # Ban MVP hallmark naming and MVP-pattern vocabulary in calcification POSITIVES
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multiple lumina", "crowded vessels",
]

MICROCYSTS_BLACKLIST = [
    # Do not let microcysts prompts drift into the two hallmark vocabularies
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "endothelial", "angiogenesis", "neovascular", "vascular hyperplasia",
    # (Optional stricter) block very hallmark-anchored tokens often used for MVP descriptions
    "tuft", "tufted", "capillary tuft",
]

EDEMA_BLACKLIST = [
    # Necrosis-pattern terms (avoid leakage into necrosis)
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "acellular", "cell dropout", "ghost", "karyorrhexis", "karyorrhectic",
    "debris", "architectural collapse", "tissue breakdown", "geographic", "featureless",
    # MVP/vascular-proliferation terms (avoid leakage into MVP)
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid", "tuft", "tufted", "capillary tuft",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    "multiple lumina", "crowded vessels",
]

REACTIVE_GLIOSIS_BLACKLIST = [
    # Competing hallmark names / near-names (explicit leakage)
    "necrosis", "necrotic", "palisading", "pseudopalisading",
    "microvascular", "vascular proliferation", "microvascular proliferation",
    "glomeruloid",
    "endothelial", "endothelial proliferation", "endothelial hyperplasia",
    "angiogenesis", "neovascular", "vascular hyperplasia",
    # Diagnosis-leaning terms (extra safety on top of GLOBAL_BLACKLIST)
    "glioma", "tumor", "neoplastic",
    "hypercellularity", "pleomorphism", "mitoses",
]

PER_EXPERT_BLACKLIST: Dict[str, List[str]] = {
    MVP_PACK.name: MVP_BLACKLIST,
    NECROSIS_PACK.name: NECROSIS_BLACKLIST,
    HEMORRHAGE_PACK.name: HEMORRHAGE_BLACKLIST,
    THROMBOSIS_PACK.name: THROMBOSIS_BLACKLIST,
    HIGH_VASCULARITY_PACK.name: HIGH_VASCULARITY_BLACKLIST,
    NORMAL_PARENCHYMA_PACK.name: NORMAL_PARENCHYMA_BLACKLIST,
    HYPERCELLULARITY_PACK.name: HYPERCELLULARITY_BLACKLIST,
    PLEOMORPHISM_PACK.name: PLEOMORPHISM_BLACKLIST,
    MITOSES_PACK.name: MITOSES_BLACKLIST,
    INFILTRATION_PACK.name: INFILTRATION_BLACKLIST,
    ARTIFACT_PACK.name: ARTIFACT_BLACKLIST,
    CALCIFICATION_PACK.name: CALCIFICATION_BLACKLIST,
    MICROCYSTS_PACK.name: MICROCYSTS_BLACKLIST,
    EDEMA_PACK.name: EDEMA_BLACKLIST,
    REACTIVE_GLIOSIS_PACK.name: REACTIVE_GLIOSIS_BLACKLIST,
}

# Hard-negative integrity blacklist (do not name the competing hallmark in hard negatives)
HARD_NEGATIVE_INTEGRITY: Dict[str, List[str]] = {
    MVP_PACK.name: [
        "necrosis", "necrotic", "palisading", "pseudopalisading"
    ],
    NECROSIS_PACK.name: [
        "microvascular proliferation", "glomeruloid", "endothelial proliferation", "vascular hyperplasia"
    ],
    HEMORRHAGE_PACK.name: [
        # Must not name competing hallmarks (hard-negative integrity)
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "glomeruloid",
        "endothelial proliferation", "vascular hyperplasia", "angiogenesis",

        # Must not include hemorrhage vocabulary (recommended)
        "hemorrhage", "hemorrhagic", "bleeding",
        "blood", "red blood cell", "rbc", "erythrocyte", "extravasated",
        "hemosiderin", "siderophage", "hematoidin", "clot",
    ],
    THROMBOSIS_PACK.name: [
        # Must not name competing hallmarks (hard-negative integrity)
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "glomeruloid",
        "endothelial proliferation", "vascular hyperplasia", "angiogenesis",
        "hemorrhage", "hemorrhagic", "bleeding",

        # Must not include thrombosis vocabulary (recommended)
        # TODO: How do you ensure complete converage of vocabulary
        "thrombosis", "thrombus", "clot", "coagulum", "embolus",
        "fibrin", "platelet", "occlusion", "occlusive",
        "intravascular", "intraluminal", "lumen-filling", "attached to vessel wall",
        "laminated", "lines of zahn", "recanalization",
    ],
    HIGH_VASCULARITY_PACK.name: [
        "microvascular proliferation", "microvascular",
        "glomeruloid",
        "tuft", "tufted",
        "endothelial proliferation",
        "vascular hyperplasia",
        "angiogenesis",
        "necrosis", "necrotic",
        "palisading", "pseudopalisading",
    ],
    HYPERCELLULARITY_PACK.name: [
        # Must not name necrosis hallmark
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        # Must not name MVP hallmark
        "microvascular proliferation", "vascular hyperplasia", "endothelial proliferation", "glomeruloid",
    ],
    NORMAL_PARENCHYMA_PACK.name: [
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "vascular proliferation",
        "glomeruloid", "endothelial proliferation", "vascular hyperplasia",
        "angiogenesis", "mvp",
    ],
    PLEOMORPHISM_PACK.name: [
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "microvascular", "vascular proliferation",
        "glomeruloid", "endothelial proliferation", "vascular hyperplasia",
        "angiogenesis", "neovascular", "endothelial",
    ],
    MITOSES_PACK.name: [
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "glomeruloid", "endothelial proliferation", "vascular hyperplasia",
        "mvp",
    ],
    INFILTRATION_PACK.name: [
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "vascular proliferation",
        "glomeruloid", "endothelial proliferation", "vascular hyperplasia",
    ],
    ARTIFACT_PACK.name: [
        "artifact", "artifactual",
        "blur", "blurry", "out-of-focus", "defocus",
        "motion", "smear", "smearing", "streak", "streaking",
        "glare", "saturated", "saturation",
        "banding", "banded", "stripe", "striping", "scanline",
        "stitch", "stitching", "seam",
        "vignetting", "vignette",
        "noise", "noisy", "speckle", "speckling",
        "dead pixel", "hot pixel",
        "compression", "blocky",
        "dust", "hair", "bubble", "contamination",
    ],
    CALCIFICATION_PACK.name: [
        # Must not name necrosis hallmark
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        # Must not name MVP hallmark
        "microvascular proliferation", "vascular proliferation",
        "glomeruloid", "endothelial proliferation", "vascular hyperplasia",
        "angiogenesis",
    ],
    MICROCYSTS_PACK.name: [
        # competing-hallmark naming bans
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "microvascular", "glomeruloid",
        "endothelial proliferation", "endothelial", "vascular hyperplasia", "angiogenesis",
    ],
    EDEMA_PACK.name: [
        # Do not name necrosis or palisading variants
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        # Do not name vascular-proliferation hallmark terms
        "microvascular proliferation", "vascular proliferation", "microvascular",
        "glomeruloid", "endothelial proliferation", "vascular hyperplasia", "angiogenesis",
    ],
    REACTIVE_GLIOSIS_PACK.name: [
        "necrosis", "necrotic", "palisading", "pseudopalisading",
        "microvascular proliferation", "vascular hyperplasia",
        "glomeruloid", "endothelial proliferation", "endothelial hyperplasia",
        "angiogenesis", "neovascular",
    ],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_violations(text: str, banned_terms: List[str]) -> List[str]:
    """
    Conservative matching:
    - For multiword terms: substring match on normalized text.
    - For single tokens: word-boundary match on normalized text.
    """
    t = _normalize(text)
    violations = []
    for term in banned_terms:
        term_n = _normalize(term)
        if not term_n:
            continue
        if " " in term_n:
            if term_n in t:
                violations.append(term)
        else:
            if re.search(r"\b" + re.escape(term_n) + r"\b", t):
                violations.append(term)
    return violations


# =============================================================================
# 3b) Anchor-based distinctiveness constraints
# =============================================================================
#
# Rationale:
# - Many concepts collapse in embedding space if prompts share a long common prefix and use generic morphology.
# - We enforce that *positives* contain at least one concept-specific anchor term early in the morphology clause,
#   and that *hard negatives* do not accidentally include those positive anchors.
#
# Notes:
# - "Early" is defined over WORDS (not CLIP tokens): we take the morphology tail (after "showing" if present),
#   then require an anchor within the first ANCHOR_WINDOW_WORDS words.
# - Anchors are automatically derived from each expert's positives_short phrases, then optionally enriched
#   with minimal expert-specific overrides.

ANCHOR_WINDOW_WORDS = int(os.environ.get("ANCHOR_WINDOW_WORDS", "12"))
ANCHOR_NGRAM_MAX = int(os.environ.get("ANCHOR_NGRAM_MAX", "3"))
ANCHOR_MAX_TERMS_PER_EXPERT = int(os.environ.get("ANCHOR_MAX_TERMS_PER_EXPERT", "60"))

_ANCHOR_STOPWORDS = {
    "a","an","and","or","the","of","to","in","on","at","for","from","by","with","without","within","across",
    "showing","image","patch","histology","microscopy","brain","tissue","pseudo","color","false","composite",
    "hsi","hyperspectral","pathology",
}

# Words that are frequently non-discriminative when used as *single-token* anchors.
_GENERIC_SINGLE_ANCHORS = {
    "cell","cells","nucleus","nuclei","nuclear","cytoplasm","tissue","field","region","area","background",
    "dense","diffuse","increased","decreased","numerous","multiple","many","small","large","uniform","irregular",
    "preserved","disrupted","structure","architecture","texture","profiles","profile","scattered","distributed",
    "crowded","cues","channels","vessel","vessels","vascular","lumina","lumen","ring","rings","pathology",
}

def _word_tokens(text: str) -> List[str]:
    # Keep hyphenated words; drop punctuation.
    return re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text.lower())

def _build_anchor_terms_from_short_phrases(shorts: List[str]) -> List[str]:
    terms = set()
    for s in shorts:
        toks = [t for t in _word_tokens(s) if t not in _ANCHOR_STOPWORDS]
        # Single tokens
        for t in toks:
            if t in _GENERIC_SINGLE_ANCHORS:
                continue
            if len(t) >= 3 and not t.isdigit():
                terms.add(t)
        # N-grams
        for n in range(2, max(2, ANCHOR_NGRAM_MAX) + 1):
            for i in range(0, len(toks) - n + 1):
                ng = " ".join(toks[i:i+n]).strip()
                if not ng:
                    continue
                # Keep ngrams even if they contain some generic tokens as long as not ALL are generic/stopwords.
                ng_words = ng.split()
                if all((w in _ANCHOR_STOPWORDS) or (w in _GENERIC_SINGLE_ANCHORS) for w in ng_words):
                    continue
                terms.add(ng)

    # Prefer longer n-grams, but keep a small reserve of single-token anchors for robustness.
    ngrams = sorted([t for t in terms if " " in t], key=lambda x: (x.count(" "), len(x)), reverse=True)
    singles = sorted([t for t in terms if " " not in t], key=len, reverse=True)
    min_singles = min(20, ANCHOR_MAX_TERMS_PER_EXPERT)
    keep_ngrams = max(0, ANCHOR_MAX_TERMS_PER_EXPERT - min_singles)
    out = ngrams[:keep_ngrams] + singles[:min_singles]
    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
    return dedup


# Minimal manual enrichment for common confusers (kept intentionally small).
# These should be MORPHOLOGY-ONLY words/phrases (no diagnosis).
EXPERT_ANCHOR_OVERRIDES: Dict[str, List[str]] = {
    "microvascular proliferation": ["glomeruloid", "tufted", "multilayered endothelium", "capillary tuft"],
    "high vascularity": ["vessel density", "not clustered", "separate vascular profiles", "isolated luminal rings"],
    "hypercellularity": ["nuclear density", "cellular crowding", "near-contiguous nuclei"],
    "pleomorphism": ["anisokaryosis", "bizarre nuclei", "multinucleated", "giant cell"],
    "infiltration cues": ["perineuronal satellitosis", "perivascular cuffing", "single-cell infiltration"],
    "artifact": ["banding", "seam", "vignetting", "speckling", "contamination"],
}

EXPERT_POS_ANCHORS: Dict[str, List[str]] = {}
for _ename, _pack in EXPERTS.items():
    _anchor_sources = list(_pack.positives_short) + list(_pack.positives)
    _anchors = _build_anchor_terms_from_short_phrases(_anchor_sources)
    _anchors += [a.strip().lower() for a in EXPERT_ANCHOR_OVERRIDES.get(_ename, []) if a.strip()]
    # Deduplicate while preserving (rough) preference for longer anchors
    _anchors = sorted(set(_anchors), key=lambda x: (x.count(" "), len(x)), reverse=True)
    EXPERT_POS_ANCHORS[_ename] = _anchors[:ANCHOR_MAX_TERMS_PER_EXPERT]

def _morphology_tail(prompt: str) -> str:
    p = prompt.strip()
    m = re.search(r"\bshowing\b\s*(.+)$", p, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if p.lower().startswith(("hsi", "hyperspectral")) and "," in p:
        return p.split(",", 1)[1].strip()
    return p

def _anchor_hits(text: str, anchors: List[str]) -> List[str]:
    # Reuse the same robust matching as blacklists.
    return _find_violations(text, anchors)

def _has_anchor_early(expert_name: str, prompt: str) -> bool:
    anchors = EXPERT_POS_ANCHORS.get(expert_name, [])
    if not anchors:
        return True  # nothing to enforce
    tail = _morphology_tail(prompt)
    words = _word_tokens(tail)
    early = " ".join(words[:ANCHOR_WINDOW_WORDS]).strip()
    return len(_anchor_hits(early, anchors)) > 0

def _contains_positive_anchor_anywhere(expert_name: str, prompt: str) -> List[str]:
    anchors = EXPERT_POS_ANCHORS.get(expert_name, [])
    if not anchors:
        return []
    # For hard-negatives, only block high-confidence positive anchors (phrases and explicit overrides).
    override = {a.strip().lower() for a in EXPERT_ANCHOR_OVERRIDES.get(expert_name, []) if a.strip()}
    block = [a for a in anchors if (" " in a) or (a in override)]
    # Ignore modality boilerplate; enforce over the morphology tail.
    return _anchor_hits(_morphology_tail(prompt), block)




def _replace_morphology_tail(prompt: str, new_tail: str) -> str:
    p = prompt.strip()
    m = re.search(r"\bshowing\b\s*", p, flags=re.IGNORECASE)
    if m:
        prefix = p[:m.end()]
        return (prefix + new_tail).strip()
    if p.lower().startswith(("hsi", "hyperspectral")) and "," in p:
        head, _rest = p.split(",", 1)
        return (head.strip() + ", " + new_tail).strip()
    return new_tail.strip()


def _frontload_required_anchor(expert_name: str, prompt: str) -> str:
    """Ensure positives carry a concept anchor within the first ANCHOR_WINDOW_WORDS words.

    Deterministic fallback (no GPT required).
    Selection policy:
      - If an explicit per-expert override exists, prefer that first (prevents choosing negated/contrast phrases).
      - Otherwise, reuse an anchor that appears somewhere in the tail.
      - Otherwise, fall back to the top-ranked anchor.
    """
    if _has_anchor_early(expert_name, prompt):
        return prompt

    tail = _morphology_tail(prompt)
    anchors = EXPERT_POS_ANCHORS.get(expert_name, [])
    if not anchors:
        return prompt

    ov = [a.strip().lower() for a in EXPERT_ANCHOR_OVERRIDES.get(expert_name, []) if a.strip()]
    if ov:
        anchor = ov[0]
    else:
        hits = _anchor_hits(tail, anchors)
        if hits:
            anchor = sorted(hits, key=lambda x: (x.count(" "), len(x)), reverse=True)[0]
        else:
            anchor = anchors[0]

    # Remove one instance if it already exists later to reduce repetition.
    tail_no = re.sub(re.escape(anchor), "", tail, flags=re.IGNORECASE, count=1)
    tail_no = re.sub(r"\s{2,}", " ", tail_no).strip(" ,;.")

    new_tail = f"{anchor}, {tail_no}" if tail_no else anchor
    return _replace_morphology_tail(prompt, new_tail)


def _strip_positive_anchors_from_hard_negative(expert_name: str, prompt: str) -> str:
    """Remove any accidental *positive* anchor phrases from hard-negatives.

    We only remove high-confidence anchors (phrases and explicit overrides) to avoid over-sanitizing.
    """
    hits = _contains_positive_anchor_anywhere(expert_name, prompt)
    if not hits:
        return prompt
    tail = _morphology_tail(prompt)
    # Remove longest hits first
    for h in sorted(set(hits), key=lambda x: (x.count(" "), len(x)), reverse=True):
        tail = re.sub(re.escape(h), "", tail, flags=re.IGNORECASE)
    tail = re.sub(r"\s{2,}", " ", tail).strip(" ,;.")
    return _replace_morphology_tail(prompt, tail)

def lint_prompt(
    expert_name: str,
    prompt: str,
    prompt_kind: str,  # "positive" or "hard_negative"
) -> List[str]:
    violations: List[str] = []

    # Must mention HSI explicitly
    # if "hsi" not in prompt.lower() and "hyperspectral" not in prompt.lower():
    #     violations.append("missing_hsi_reference")

    # Positives must contain at least one concept-specific anchor early in the morphology tail
    if prompt_kind == "positive":
        if not _has_anchor_early(expert_name, prompt):
            violations.append(f"missing_anchor_early:first_{ANCHOR_WINDOW_WORDS}_words")

    # Hard negatives must NOT contain the expert's own positive anchors (prevents neg collapsing into pos)
    if prompt_kind == "hard_negative":
        hits = _contains_positive_anchor_anywhere(expert_name, prompt)
        if hits:
            # include up to 3 hits for debugging
            violations.append("hard_negative_contains_positive_anchor:" + ",".join(hits[:3]))

    # Global blacklist always applies
    violations += [f"global:{v}" for v in _find_violations(prompt, GLOBAL_BLACKLIST)]

    if prompt_kind == "positive":
        # Expert-specific blacklist applies only to positives (prevents cross-concept leakage)
        expert_terms = PER_EXPERT_BLACKLIST.get(expert_name, [])
        violations += [f"expert:{v}" for v in _find_violations(prompt, expert_terms)]

    elif prompt_kind == "hard_negative":
        # Hard-negative integrity applies only to hard negatives (prevents naming competing hallmark)
        hn_terms = HARD_NEGATIVE_INTEGRITY.get(expert_name, [])
        violations += [f"hn_integrity:{v}" for v in _find_violations(prompt, hn_terms)]

    else:
        violations.append("invalid_prompt_kind")

    return violations


# =============================================================================
# 4) Optional: Expand prompt banks using GPT-5.2 (thinking) with strict JSON and repair loop
# =============================================================================

GEN_SYSTEM_SPEC = f"""
You generate text prompts for a medical CLIP-style text encoder.

Hard constraints:
- Return valid JSON only. No markdown, no extra keys.
- Each prompt must be 1 to 2 sentences and describe visual appearance only (no diagnosis, no grading, no molecular markers).
- Do not use spectral theory claims (no wavelength peaks, absorption bands).
- Do not mention correlated duplicates (no palisading expert; do not introduce it).
- Avoid any banned terms provided in the input.

Output schema:
{{
  "expert": string,
  "positives": [string, ...],
  "hard_negatives": [string, ...]
}}
"""


def _require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install --upgrade openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")
    return OpenAI(api_key=api_key)


def generate_variants_with_gpt(
    expert_name: str,
    n_pos: int,
    n_neg: int,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    vector_store_id: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Generates *additional* prompt variants for a given expert, then lints them.
    If vector_store_id is provided, the model can use file_search over your vector store
    to ground morphology wording.
    """
    client = _require_openai_client()

    payload = {
        "expert": expert_name,
        "counts": {"positives": n_pos, "hard_negatives": n_neg},
        # "fixed_modality_phrase": MODALITY_PHRASE,
        "global_blacklist": GLOBAL_BLACKLIST,
        "expert_blacklist": PER_EXPERT_BLACKLIST.get(expert_name, []),
        "hard_negative_integrity_blacklist": HARD_NEGATIVE_INTEGRITY.get(expert_name, []),
        "anchor_window_words": ANCHOR_WINDOW_WORDS,
        "required_anchor_terms": EXPERT_POS_ANCHORS.get(expert_name, [])[:25],
        "seed_examples": {
            "positives": EXPERTS[expert_name].positives,
            "hard_negatives": EXPERTS[expert_name].hard_negatives,
        },
        "instructions": [
            "If file_search is available, first retrieve short morphology descriptors for this expert concept from the knowledge base.",
            "Use only morphology wording; do not quote long passages.",
            "Do not add diagnosis, grade, or molecular terms.",
            "For positives: focus tightly on the target morphology for this expert.",
            "For hard_negatives: describe the competing morphology without naming it.",
        ],
    }

    tools = []
    if vector_store_id:
        tools.append({"type": "file_search", "vector_store_ids": [vector_store_id]})

    resp = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        tools=tools if tools else None,
        input=[
            {"role": "system", "content": GEN_SYSTEM_SPEC},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )

    data = json.loads(resp.output_text)
    print(f"{expert_name = }")

    return {
        "positives": [str(x).strip() for x in data.get("positives", []) if str(x).strip()],
        "hard_negatives": [str(x).strip() for x in data.get("hard_negatives", []) if str(x).strip()],
    }



def repair_prompt_with_gpt(
    expert_name: str,
    prompt: str,
    prompt_kind: str,
    violations: List[str],
    model: str = "gpt-5.2",
) -> str:
    """
    Attempts to rewrite a single prompt to satisfy lint rules while keeping intended meaning.
    Returns the rewritten prompt text (not JSON).
    """
    client = _require_openai_client()

    repair_spec = {
        "expert": expert_name,
        "prompt_kind": prompt_kind,
        # "fixed_modality_phrase": MODALITY_PHRASE,
        "original_prompt": prompt,
        "violations": violations,
        "global_blacklist": GLOBAL_BLACKLIST,
        "expert_blacklist": PER_EXPERT_BLACKLIST.get(expert_name, []),
        "hard_negative_integrity_blacklist": HARD_NEGATIVE_INTEGRITY.get(expert_name, []) if prompt_kind == "hard_negative" else [],
        "anchor_window_words": ANCHOR_WINDOW_WORDS,
        "required_anchor_terms": EXPERT_POS_ANCHORS.get(expert_name, [])[:25],
        "rewrite_rules": [
            "Return only the rewritten prompt text. No quotes, no JSON.",
            "Keep 1 to 2 sentences.",
            # "Must include 'HSI' or 'Hyperspectral'. Prefer using the fixed modality phrase verbatim.",
            "Do not add diagnosis, grade, or molecular terms.",
            "Do not introduce the competing hallmark name.",
            "For POSITIVE prompts: include at least one required_anchor_terms phrase and place it within the first anchor_window_words words of the morphology clause.",
        ],
    }

    resp = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": "Rewrite the prompt to satisfy the constraints. Output only the rewritten prompt text."},
            {"role": "user", "content": json.dumps(repair_spec)},
        ],
    )
    return resp.output_text.strip()


def lint_and_repair_bank(
    expert_name: str,
    positives: List[str],
    hard_negatives: List[str],
    max_repair_rounds: int = 8,
    allow_gpt_repair: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Lints all prompts. If allow_gpt_repair is True, attempts to repair violating prompts.
    Raises ValueError if a prompt cannot be repaired.
    """
    fixed_pos: List[str] = []
    for p in positives:
        p0 = p.strip()
        violations = lint_prompt(expert_name, p0, "positive")
        if violations and allow_gpt_repair:
            cur = p0
            print(f"Found positive violations with {cur}. Attempting repair for {expert_name} expert...")
            for _ in range(max_repair_rounds):
                cur = repair_prompt_with_gpt(expert_name, cur, "positive", violations)
                violations = lint_prompt(expert_name, cur, "positive")
                if not violations:
                    break
            if violations:
                raise ValueError(f"Unrepairable positive prompt for {expert_name}: {violations} | {p0}")
            fixed_pos.append(cur)
        elif violations:
            raise ValueError(f"Positive prompt failed lint for {expert_name}: {violations} | {p0}")
        else:
            fixed_pos.append(p0)

    fixed_neg: List[str] = []
    for p in hard_negatives:
        p0 = p.strip()
        violations = lint_prompt(expert_name, p0, "hard_negative")
        if violations and allow_gpt_repair:
            cur = p0
            print(f"Found negative violation with {cur}. Attempting repair for {expert_name} expert...")
            for _ in range(max_repair_rounds):
                cur = repair_prompt_with_gpt(expert_name, cur, "hard_negative", violations)
                violations = lint_prompt(expert_name, cur, "hard_negative")
                if not violations:
                    break
            if violations:
                raise ValueError(f"Unrepairable hard-negative prompt for {expert_name}: {violations} | {p0}")
            fixed_neg.append(cur)
        elif violations:
            raise ValueError(f"Hard-negative prompt failed lint for {expert_name}: {violations} | {p0}")
        else:
            fixed_neg.append(p0)

    return fixed_pos, fixed_neg


# =============================================================================
# 4.5) CLIP token-budget enforcement (prompt-time, not embed-time)
# =============================================================================

_CACHED_CLIP_TOKENIZER: Optional[object] = None
_CACHED_CLIP_MAXLEN: Optional[int] = None

def _get_clip_tokenizer_and_max_len() -> Tuple[object, int]:
    """Return (tokenizer, max_len) for the configured CLIP text model.

    Uses CLIP_TEXT_MODEL_ID and CLIP_MAX_TOKENS defaults. Falls back to
    whitespace budgeting if transformers is unavailable.

    Note: CLIP max length includes special tokens. For PLIP/CLIP this is typically 77.
    """
    global _CACHED_CLIP_TOKENIZER, _CACHED_CLIP_MAXLEN
    if _CACHED_CLIP_TOKENIZER is not None and _CACHED_CLIP_MAXLEN is not None:
        return _CACHED_CLIP_TOKENIZER, _CACHED_CLIP_MAXLEN

    if AutoTokenizer is None:
        return None, CLIP_MAX_TOKENS  # type: ignore

    max_len = CLIP_MAX_TOKENS
    if AutoConfig is not None:
        try:
            cfg = AutoConfig.from_pretrained(CLIP_TEXT_MODEL_ID)
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None and getattr(text_cfg, "max_position_embeddings", None) is not None:
                max_len = int(text_cfg.max_position_embeddings)
            elif getattr(cfg, "max_position_embeddings", None) is not None:
                max_len = int(cfg.max_position_embeddings)
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(CLIP_TEXT_MODEL_ID, use_fast=True)
    _CACHED_CLIP_TOKENIZER = tok
    _CACHED_CLIP_MAXLEN = max_len
    return tok, max_len


def _clip_token_len(tokenizer: object, text: str) -> int:
    """CLIP token length including special tokens (or whitespace fallback)."""
    if tokenizer is None:
        return len(text.split())
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]  # type: ignore[index]
    return int(len(ids))


def _canonicalize_modality_prefix(prompt: str) -> str:
    """Compress verbose modality prefixes into the single fixed MODALITY_PHRASE."""
    return prompt
    p = prompt.strip()

    # If we can find a "..." clause, rewrite as "{MODALITY_PHRASE}, ..."
    m = re.search(r"\bshowing\b\s*(.+)$", p, flags=re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        return f"{MODALITY_PHRASE}, {tail}"

    # Otherwise, if the prompt begins with some HSI/Hyperspectral prefix, keep content after the first comma.
    if p.lower().startswith(("hsi", "hyperspectral")) and "," in p:
        tail = p.split(",", 1)[1].strip()
        return f"{MODALITY_PHRASE}, {tail}"

    return p


def _ensure_prompt_leq_max_tokens(prompt: str, tokenizer: object, max_len: int) -> Tuple[str, bool]:
    """Ensure a prompt is <= max_len CLIP tokens. Returns (prompt, changed)."""
    original = prompt.strip()
    p = _canonicalize_modality_prefix(original)

    # Ensure modality marker exists (lint requires HSI or Hyperspectral)
    # if ("HSI" not in p) and ("Hyperspectral" not in p):
    #     p = f"{MODALITY_PHRASE}, {p}"

    p = re.sub(r"\s+", " ", p).strip()
    did_modify = (p != original)

    if _clip_token_len(tokenizer, p) <= max_len:
        return p, did_modify

    trimmed = p

    # Iterative clause trimming (preserve the beginning where the concept should be)
    for _ in range(12):
        if _clip_token_len(tokenizer, trimmed) <= max_len:
            return trimmed, True

        if "," in trimmed:
            trimmed = trimmed.rsplit(",", 1)[0].strip()
        elif ";" in trimmed:
            trimmed = trimmed.rsplit(";", 1)[0].strip()
        elif " with " in trimmed.lower():
            parts = re.split(r"\bwith\b", trimmed, flags=re.IGNORECASE)
            trimmed = parts[0].strip()
        else:
            words = trimmed.split()
            if len(words) <= 12:
                break
            trimmed = " ".join(words[:-6]).strip()

        trimmed = re.sub(r"\s+", " ", trimmed).strip()
        if not trimmed.endswith("."):
            trimmed = trimmed + "."

    # Last resort: tokenizer truncation + decode (guarantees <= max_len if tokenizer exists)
    if tokenizer is not None:
        enc = tokenizer(trimmed, add_special_tokens=True, truncation=True, max_length=max_len)
        ids = enc["input_ids"]  # type: ignore[index]
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        decoded = re.sub(r"\s+", " ", decoded).strip()
        # if ("HSI" not in decoded) and ("Hyperspectral" not in decoded):
        #     decoded = f"{MODALITY_PHRASE}, {decoded}"
        return decoded, True

    # Fallback: whitespace truncate
    words = trimmed.split()
    decoded = " ".join(words[:max_len]).strip()
    # if ("HSI" not in decoded) and ("Hyperspectral" not in decoded):
    #     decoded = f"{MODALITY_PHRASE}, {decoded}"
    return decoded, True


def enforce_clip_token_budget(
    expert_name: str,
    positives: List[str],
    hard_negatives: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Enforce CLIP token budget on both positives and hard negatives.

    Returns (positives_trimmed, hard_negatives_trimmed, reports). Reports contain
    before/after strings for any prompt that was modified.
    """
    tokenizer, max_len = _get_clip_tokenizer_and_max_len()
    reports: List[str] = []

    # Enforce anchor placement BEFORE trimming to avoid deleting discriminative clauses.
    positives = [_frontload_required_anchor(expert_name, p) for p in positives]
    hard_negatives = [_strip_positive_anchors_from_hard_negative(expert_name, p) for p in hard_negatives]

    pos_out: List[str] = []
    for p in positives:
        p2, changed = _ensure_prompt_leq_max_tokens(p, tokenizer, max_len)
        if changed:
            reports.append(f"[{expert_name}][positive] trimmed_to_{max_len}_tokens: {p!r} -> {p2!r}")
        pos_out.append(p2)

    neg_out: List[str] = []
    for p in hard_negatives:
        p2, changed = _ensure_prompt_leq_max_tokens(p, tokenizer, max_len)
        if changed:
            reports.append(f"[{expert_name}][hard_negative] trimmed_to_{max_len}_tokens: {p!r} -> {p2!r}")
        neg_out.append(p2)

    # Re-lint after trimming to guarantee invariants.
    pos_final: List[str] = []
    for p in pos_out:
        v = lint_prompt(expert_name, p, "positive")
        if v:
            raise ValueError(f"Positive prompt failed lint AFTER token-trim for {expert_name}: {v} | {p!r}")
        pos_final.append(p)

    neg_final: List[str] = []
    for p in neg_out:
        v = lint_prompt(expert_name, p, "hard_negative")
        if v:
            raise ValueError(f"Hard-negative prompt failed lint AFTER token-trim for {expert_name}: {v} | {p!r}")
        neg_final.append(p)

    return pos_final, neg_final, reports


# =============================================================================
# 5) Pack assembly + JSON export
# =============================================================================

def build_prompt_pack(
    expand_with_gpt: bool = False,
    extra_pos_per_expert: int = 0,
    extra_neg_per_expert: int = 0,
    vector_store_id: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Builds the complete prompt pack with linting.
    If expand_with_gpt is True, generates additional variants and repairs if needed.
    """
    pack: Dict[str, Dict[str, List[str]]] = {
        "meta": {
            # "modality_phrase": MODALITY_PHRASE,
            "experts": list(EXPERTS.keys()),
            "note": "",
        },
        "blacklists": {
            "global": GLOBAL_BLACKLIST,
            "per_expert": PER_EXPERT_BLACKLIST,
            "hard_negative_integrity": HARD_NEGATIVE_INTEGRITY,
        },
        "experts": {},
    }

    token_trim_reports_all: List[str] = []

    for expert_name, expert in EXPERTS.items():
        # Seed prompts (already include HSI/Hyperspectral in your pack)
        pos_seed = [p.strip() for p in expert.positives] + [
            _prefix_with_modality(s.strip()) for s in expert.positives_short
        ]
        neg_seed = [p.strip() for p in expert.hard_negatives] + [
            _prefix_with_modality(s.strip()) for s in expert.hard_negatives_short
        ]

        # Optional: expand with GPT
        pos_extra: List[str] = []
        neg_extra: List[str] = []
        if expand_with_gpt and (extra_pos_per_expert > 0 or extra_neg_per_expert > 0):
            gen = generate_variants_with_gpt(
                expert_name=expert_name,
                n_pos=extra_pos_per_expert,
                n_neg=extra_neg_per_expert,
                vector_store_id=vector_store_id,
            )
            pos_extra = gen["positives"]
            neg_extra = gen["hard_negatives"]

        positives = pos_seed + pos_extra
        hard_negatives = neg_seed + neg_extra

        # Lint + (optional) repair
        # Deterministic anchor enforcement (does not require GPT repair)
        positives = [_frontload_required_anchor(expert_name, p) for p in positives]
        hard_negatives = [_strip_positive_anchors_from_hard_negative(expert_name, p) for p in hard_negatives]
        positives_fixed, hard_negatives_fixed = lint_and_repair_bank(
            expert_name=expert_name,
            positives=positives,
            hard_negatives=hard_negatives,
            allow_gpt_repair=expand_with_gpt,  # only repair if GPT is enabled
        )

        pack["experts"][expert_name] = {
            "positives": positives_fixed,
            "hard_negatives": hard_negatives_fixed,
        }

    # Record token trimming summary (kept within schema as meta.note)
    try:
        _, inferred_max_len = _get_clip_tokenizer_and_max_len()
    except Exception:
        inferred_max_len = CLIP_MAX_TOKENS
    if token_trim_reports_all:
        pack["meta"]["note"] = (
            f"Enforced <= {inferred_max_len} CLIP text tokens using '{CLIP_TEXT_MODEL_ID}'. "
            f"Trimmed {len(token_trim_reports_all)} prompts."
        )
    else:
        pack["meta"]["note"] = (
            f"Enforced <= {inferred_max_len} CLIP text tokens using '{CLIP_TEXT_MODEL_ID}'. No trims required."
        )

    return pack


def save_prompt_pack(pack: Dict[str, Dict[str, List[str]]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False)


def audit_prompts(
    require_exact_modality_phrase: bool = False,
    ban_other_expert_names: bool = False,
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    """
    Returns a complete audit report without raising.
    Does not change any output JSON schema.
    """
    report: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

    expert_names = list(EXPERTS.keys())

    def other_expert_hits(current: str, prompt: str) -> List[str]:
        t = prompt.lower()
        hits = []
        for name in expert_names:
            if name == current:
                continue
            if name.lower() in t:
                hits.append(name)
        return hits

    for expert_name, expert in EXPERTS.items():
        pos = [p.strip() for p in expert.positives] + [_prefix_with_modality(s.strip()) for s in expert.positives_short]
        neg = [p.strip() for p in expert.hard_negatives] + [_prefix_with_modality(s.strip()) for s in expert.hard_negatives_short]

        # Audit should reflect the *exported* prompt strings:
        # - enforce early-anchor constraint on positives
        # - strip positive anchors from hard-negatives
        # - (best-effort) enforce CLIP token budget if tokenizer is available
        pos = [_frontload_required_anchor(expert_name, p) for p in pos]
        neg = [_strip_positive_anchors_from_hard_negative(expert_name, p) for p in neg]
        try:
            tokenizer, max_len = _get_clip_tokenizer_and_max_len()
            pos = [_ensure_prompt_leq_max_tokens(p, tokenizer, max_len)[0] for p in pos]
            neg = [_ensure_prompt_leq_max_tokens(p, tokenizer, max_len)[0] for p in neg]
        except Exception:
            # If tokenizer isn't available, still enforce anchor and leakage rules above.
            pass

        expert_report = {"positives": [], "hard_negatives": []}

        for p in pos:
            violations = lint_prompt(expert_name, p, "positive")

            # if require_exact_modality_phrase and MODALITY_PHRASE.lower() not in p.lower():
            #     violations.append("missing_exact_modality_phrase")

            if ban_other_expert_names:
                hits = other_expert_hits(expert_name, p)
                if hits:
                    violations.append(f"mentions_other_expert_names:{hits}")

            if violations:
                expert_report["positives"].append({"prompt": p, "violations": violations})

        for p in neg:
            violations = lint_prompt(expert_name, p, "hard_negative")

            # if require_exact_modality_phrase and MODALITY_PHRASE.lower() not in p.lower():
            #     violations.append("missing_exact_modality_phrase")

            if ban_other_expert_names:
                hits = other_expert_hits(expert_name, p)
                if hits:
                    violations.append(f"mentions_other_expert_names:{hits}")

            if violations:
                expert_report["hard_negatives"].append({"prompt": p, "violations": violations})

        if expert_report["positives"] or expert_report["hard_negatives"]:
            report[expert_name] = expert_report

    return report


def print_audit_report(report: Dict[str, Dict[str, List[Dict[str, object]]]]) -> None:
    if not report:
        print("AUDIT PASS: no violations found.")
        return

    print("AUDIT FAIL: violations found.")
    for expert_name, buckets in report.items():
        print(f"\nEXPERT: {expert_name}")
        for kind in ["positives", "hard_negatives"]:
            items = buckets.get(kind, [])
            if not items:
                continue
            print(f"  {kind}: {len(items)} issue(s)")
            for item in items[:5]:
                print(f"    - violations: {item['violations']}")
                print(f"      prompt: {item['prompt']}")

    sys.exit(-1)


# =============================================================================
# 6) Minimal CLI usage
# =============================================================================

if __name__ == "__main__":

    # Optional audit mode before export
    report = audit_prompts(
        require_exact_modality_phrase=False,
        ban_other_expert_names=True,
    )
    print_audit_report(report)

    # Default: just validate and export the exact seed pack.
    # Set expand_with_gpt=True to generate more variants using GPT-5.2 thinking model.
    prompt_pack = build_prompt_pack(
        expand_with_gpt=True,
        extra_pos_per_expert=10,
        extra_neg_per_expert=10,
        vector_store_id='vs_6980fdd441688191ac026b6c0c69964e',
    )
    save_prompt_pack(prompt_pack, "hsi_concept_prompt_pack-10_prototypes.json")
    print("Wrote: hsi_concept_prompt_pack.json")

    



