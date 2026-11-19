"""Composting knowledge base used to ground the LLM's advice (lightweight RAG).

The CNN predicts the stage; the matching entry below is injected into the
prompt so Gemma explains and advises from real composting guidance instead of
making things up.
"""

STAGE_KNOWLEDGE = {
    "1_fresh_waste": {
        "summary": "Raw organic waste: vegetable/fruit peels, food scraps, leaves. "
                   "Original shapes and colours clearly visible. Decomposition has barely started.",
        "process": "Mesophilic phase is starting. Microbes (20-40 C) begin breaking down sugars and starches.",
        "typical_days": "Day 0-7",
        "actions": [
            "Chop or shred large pieces to increase surface area.",
            "Balance greens (food scraps, nitrogen) with browns (dry leaves, cardboard, carbon) at roughly 1 part green : 2-3 parts brown by volume (target C:N about 25-30:1).",
            "Keep moisture like a wrung-out sponge (about 50-60%).",
            "Mix well and ensure air can reach the pile.",
        ],
        "warnings": "Avoid adding meat, dairy or oily food in a home system: they cause odour and attract pests.",
        "next_stage_eta": "Visible softening and colour change usually within 1-2 weeks.",
    },
    "2_early_decomposition": {
        "summary": "Waste is softening, browning and losing shape. Some mould or white fungal growth may appear. "
                   "Pieces are still identifiable.",
        "process": "Microbial activity rises quickly; the pile may heat up and enter the thermophilic phase (40-65 C).",
        "typical_days": "Week 1-3",
        "actions": [
            "Turn or aerate the pile every 3-5 days to supply oxygen.",
            "If it smells of ammonia, add more browns (dry leaves, shredded paper).",
            "If it smells rotten or looks slimy, it is too wet or compacted: add browns and turn it.",
            "If it is dry and not heating up, sprinkle water and add some fresh greens.",
        ],
        "warnings": "White fungal threads are normal and healthy. Black slimy patches mean anaerobic conditions.",
        "next_stage_eta": "Becomes dark and mixed (active stage) in about 2-4 weeks.",
    },
    "3_active_decomposition": {
        "summary": "Dark brown, moist, mixed material. Most original items have broken down, "
                   "but some fragments (stems, peels, eggshells) are still visible.",
        "process": "Thermophilic to cooling phase. Hot microbes kill pathogens and weed seeds; "
                   "as food runs out the pile cools and fungi and worms move in.",
        "typical_days": "Week 3-8",
        "actions": [
            "Keep turning weekly while the pile is still warm.",
            "Maintain moisture; do not let it dry out completely.",
            "Stop adding fresh waste to this batch so it can finish.",
            "Remove or break up large undecomposed pieces.",
        ],
        "warnings": "Do not apply to plants yet: immature compost can harm roots and lock up nitrogen.",
        "next_stage_eta": "Matures in about 3-6 more weeks of curing.",
    },
    "4_mature_compost": {
        "summary": "Dark brown to black, crumbly and soil-like, with an earthy smell. "
                   "Original materials cannot be identified.",
        "process": "Curing/maturation phase complete. Organic matter has stabilised into humus.",
        "typical_days": "Week 8-16+",
        "actions": [
            "Sieve it to remove leftover large pieces (put them back in a new pile).",
            "Use it as soil conditioner: mix 20-30% into potting soil or spread 2-5 cm on garden beds.",
            "Store it covered and slightly moist if not using it immediately.",
            "Optional: a simple seed germination test confirms maturity.",
        ],
        "warnings": "If it still smells sour or ammonia-like, let it cure for another 2-3 weeks.",
        "next_stage_eta": "Ready to use. This is the 'wealth' from waste.",
    },
}


def stage_context(class_name: str) -> str:
    k = STAGE_KNOWLEDGE[class_name]
    actions = "\n".join(f"- {a}" for a in k["actions"])
    return (
        f"Description: {k['summary']}\n"
        f"Process: {k['process']}\n"
        f"Typical timing: {k['typical_days']}\n"
        f"Recommended actions:\n{actions}\n"
        f"Warnings: {k['warnings']}\n"
        f"Next stage: {k['next_stage_eta']}"
    )
