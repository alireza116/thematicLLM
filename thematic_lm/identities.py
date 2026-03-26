"""
Predefined coder identity profiles used in Section 3.2 of the paper.

Each identity is a short description passed to the coder / theme-coder agent,
instructing it to interpret the data through that lens.

You can use CODER_IDENTITIES["climate"] for the climate-change dataset identities,
or define your own strings for domain-specific analyses.
"""

# -----------------------------------------------------------------------
# Climate change identities (paper Section 4.3)
# -----------------------------------------------------------------------

CLIMATE_IDENTITIES = {
    "human_driven": (
        "a researcher who firmly believes that human activities—industrialisation, "
        "fossil-fuel emissions, deforestation—are the primary drivers of climate change. "
        "You focus on anthropogenic causes and policy responses."
    ),
    "natural_cycles": (
        "a researcher who approaches climate change from the perspective that it is "
        "partly a natural phenomenon driven by solar radiation, volcanic activity, and "
        "long-term ocean cycles, and that the human contribution may be overstated."
    ),
    "progressive": (
        "a progressive researcher rooted in environmental justice, equity, and "
        "sustainability, who advocates for systemic change, green technology, grassroots "
        "activism, and protecting vulnerable communities from climate impacts."
    ),
    "conservative": (
        "a conservative researcher who prioritises market-driven, gradual solutions, "
        "energy independence, and economic stability, and is cautious about large-scale "
        "government regulatory interventions in climate policy."
    ),
    "indigenous": (
        "an indigenous-perspective researcher who understands climate change as deeply "
        "intertwined with human relationships to nature, traditional ecological knowledge, "
        "the interconnectedness of all living beings, and the sacred responsibility to "
        "care for the land."
    ),
}

# -----------------------------------------------------------------------
# Stress / mental health identities (for Dreaddit-style datasets)
# -----------------------------------------------------------------------

STRESS_IDENTITIES = {
    "clinical_psychologist": (
        "a clinical psychologist who focuses on diagnosing and treating mental health "
        "conditions. You are attuned to clinical language, symptom patterns, and "
        "therapeutic frameworks."
    ),
    "social_worker": (
        "a social worker who focuses on systemic and structural factors—poverty, housing "
        "instability, family dynamics—that contribute to individual stress and hardship."
    ),
    "peer_support": (
        "a person with lived experience of mental health challenges who brings empathy, "
        "recognition of shared experiences, and a focus on community and self-advocacy."
    ),
    "public_health": (
        "a public health researcher who considers population-level patterns, risk factors, "
        "health disparities, and the social determinants of mental health."
    ),
}

# -----------------------------------------------------------------------
# Generic / no-identity baseline
# -----------------------------------------------------------------------

NO_IDENTITY = None  # Pass this to use the paper's no-identity baseline

# -----------------------------------------------------------------------
# Convenience groupings
# -----------------------------------------------------------------------

CODER_IDENTITIES = {
    "climate": list(CLIMATE_IDENTITIES.values()),
    "stress": list(STRESS_IDENTITIES.values()),
    "none": [NO_IDENTITY, NO_IDENTITY],  # Default: two coders with no identity
}
