#!/usr/bin/env python3
"""Rule-based lexicons for EVENT and RELATIONSHIP_REF subtype classification."""

import re

EVENT_LIFE_KEYWORDS = {
    "wedding", "funeral", "graduation", "breakup", "divorce", "birth", "death",
    "surgery", "diagnosis", "pregnancy", "miscarriage", "engagement", "retirement",
    "adoption", "baptism", "promotion", "accident", "anniversary", "suicide",
    "birthday", "childbirth", "hospitalization", "rehab", "relapse", "overdose",
    "proposal", "separation", "abortion", "stillbirth", "affair",
}

EVENT_GENERAL_KEYWORDS = {
    "concert", "festival", "party", "meeting", "conference", "ceremony", "prom",
    "tournament", "championship", "olympics", "game", "match", "holiday",
    "christmas", "halloween", "thanksgiving", "easter", "ramadan", "diwali",
    "reunion", "parade", "election", "barbecue", "dinner", "feast", "celebration",
    "competition", "convention", "gala", "carnival", "homecoming", "retreat",
    "workshop", "seminar", "recital", "reception", "brunch", "housewarming",
    "sleepover", "potluck", "camping", "roadtrip", "trip", "vacation", "picnic",
    "hangout",
}

EVENT_FALSE_POSITIVES = {
    "death star", "death metal", "death note", "death row", "death penalty",
    "game of thrones", "hunger games",
}

_STRIP_PREFIXES = re.compile(r"^(?:the|a|my|his|her|our|their)\s+", re.IGNORECASE)

REL_FAMILY = {
    "mom", "dad", "mother", "father", "mama", "papa", "mum", "mummy", "mommy",
    "daddy", "sister", "brother", "son", "daughter", "grandma", "grandpa",
    "grandmother", "grandfather", "granny", "nana", "nanna", "aunt", "uncle",
    "cousin", "niece", "nephew", "sibling", "twin", "parent", "parents",
    "grandparent", "grandparents", "stepmom", "stepdad", "stepmother",
    "stepfather", "stepsister", "stepbrother", "stepson", "stepdaughter",
    "half-sister", "half-brother", "mother-in-law", "father-in-law",
    "sister-in-law", "brother-in-law", "son-in-law", "daughter-in-law",
    "in-laws", "godmother", "godfather", "foster mom", "foster dad",
    "foster parent",
}

REL_ROMANTIC = {
    "boyfriend", "girlfriend", "husband", "wife", "spouse", "partner",
    "fiancé", "fiancée", "fiance", "fiancee", "ex", "ex-boyfriend",
    "ex-girlfriend", "ex-husband", "ex-wife", "lover", "significant other",
    "soulmate", "bae", "boo", "crush",
}

REL_FRIEND = {
    "friend", "best friend", "bestie", "buddy", "pal", "mate", "bff", "homie",
    "roommate", "flatmate", "housemate", "pen pal", "confidant", "confidante",
}

REL_PROFESSIONAL = {
    "boss", "manager", "coworker", "colleague", "supervisor", "mentor",
    "teacher", "professor", "doctor", "therapist", "counselor", "counsellor",
    "coach", "trainer", "tutor", "instructor", "advisor", "adviser", "nurse",
    "dentist", "lawyer", "attorney", "accountant", "principal", "dean",
}

REL_ACQUAINTANCE = {
    "neighbor", "neighbour", "acquaintance", "landlord", "landlady",
    "classmate", "schoolmate", "teammate", "penmate",
}

_REL_LEXICONS = [
    ("RELATIONSHIP_REF/Family", REL_FAMILY),
    ("RELATIONSHIP_REF/Romantic", REL_ROMANTIC),
    ("RELATIONSHIP_REF/Friend", REL_FRIEND),
    ("RELATIONSHIP_REF/Professional", REL_PROFESSIONAL),
    ("RELATIONSHIP_REF/Acquaintance", REL_ACQUAINTANCE),
]


def classify_event_by_lexicon(surface: str) -> str | None:
    stripped = _STRIP_PREFIXES.sub("", surface.lower()).strip()

    if stripped in EVENT_FALSE_POSITIVES:
        return None

    for kw in EVENT_LIFE_KEYWORDS:
        if kw in stripped:
            return "EVENT/Life"

    for kw in EVENT_GENERAL_KEYWORDS:
        if kw in stripped:
            return "EVENT/General"

    return None


def _strip_rel_surface(surface: str) -> str:
    s = surface.lower()
    for prefix in ("my ", "our ", "his ", "her ", "their ", "your "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):]
            break
    if s.endswith("'s"):
        s = s[:-2]
    return s.strip()


def classify_rel_by_lexicon(surface: str) -> str | None:
    stripped = _strip_rel_surface(surface)

    for subtype, lexicon in _REL_LEXICONS:
        if stripped in lexicon:
            return subtype
        for entry in lexicon:
            if " " in entry:
                if entry in stripped or stripped == entry:
                    return subtype
            else:
                if entry in stripped.split():
                    return subtype

    return None
