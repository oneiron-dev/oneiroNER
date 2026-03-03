#!/usr/bin/env python3
"""Re-map entity types in green dataset results using verified Sonnet mappings."""

import json
from collections import Counter, defaultdict
from pathlib import Path

REPORT_DIR = Path("docs/research")
RESULTS_DIR = REPORT_DIR / "green_results"

ONEIRON_TYPES = [
    "PERSON", "PLACE", "ORG", "DATE", "EMOTION", "GOAL",
    "ACTIVITY", "RELATIONSHIP_REF", "EVENT",
]

CROSSNER_LABELS = {
    "4": "I-album", "6": "I-algorithm", "7": "B-astronomicalobject",
    "9": "B-award", "10": "I-award", "11": "B-band", "12": "I-band",
    "14": "I-book", "21": "B-country", "25": "B-election", "26": "I-election",
    "30": "I-event", "35": "B-location", "36": "I-location",
    "41": "B-misc", "42": "I-misc", "43": "B-musicalartist",
    "49": "B-organisation", "50": "I-organisation", "51": "B-person",
    "52": "I-person", "55": "B-politicalparty", "56": "I-politicalparty",
    "57": "B-politician", "58": "I-politician", "67": "B-scientist",
    "68": "I-scientist", "76": "I-university", "77": "B-writer", "78": "I-writer",
}

VERIFIED_MAPPINGS = {
    "Broad Twitter Corpus": {
        "PER": "PERSON", "LOC": "PLACE", "ORG": "ORG",
    },
    "CoNLL 2025 NER": {
        "ORG": "ORG", "DATE": "DATE", "PERSON": "PERSON", "GPE": "PLACE",
        "MONEY": "SKIP", "CARDINAL": "SKIP", "NORP": "SKIP", "PERCENT": "SKIP",
        "WORK_OF_ART": "SKIP", "LOC": "PLACE", "TIME": "DATE",
        "QUANTITY": "SKIP", "EVENT": "EVENT", "FAC": "PLACE", "ORDINAL": "SKIP",
        "PRODUCT": "SKIP", "LAW": "SKIP", "LANGUAGE": "SKIP",
    },
    "Few-NERD": {
        "location-GPE": "PLACE", "person-other": "PERSON",
        "organization-other": "ORG", "organization-company": "ORG",
        "building-other": "PLACE", "organization-education": "ORG",
        "person-artist/author": "PERSON", "person-athlete": "PERSON",
        "person-politician": "PERSON", "organization-sportsteam": "ORG",
        "location-other": "PLACE", "event-sportsevent": "EVENT",
        "organization-government/governmentagency": "ORG",
        "location-road/railway/highway/transit": "PLACE",
        "other-award": "SKIP", "art-writtenart": "SKIP",
        "product-other": "SKIP", "art-music": "SKIP",
        "person-actor": "PERSON", "event-attack/battle/war/militaryconflict": "EVENT",
        "event-other": "EVENT", "other-biologything": "SKIP",
        "art-film": "SKIP", "location-bodiesofwater": "PLACE",
        "organization-media/newspaper": "ORG",
        "organization-politicalparty": "ORG",
        "organization-sportsleague": "ORG",
        "other-law": "SKIP", "art-other": "SKIP",
        "art-broadcastprogram": "SKIP",
        "product-airplane": "SKIP", "product-software": "SKIP",
        "product-game": "SKIP", "person-director": "PERSON",
        "person-scholar": "PERSON", "building-library": "PLACE",
        "building-airport": "PLACE", "building-hospital": "PLACE",
        "building-hotel": "PLACE", "building-restaurant": "PLACE",
        "building-sportsfacility": "PLACE", "building-theater": "PLACE",
        "location-island": "PLACE", "location-mountain": "PLACE",
        "person-soldier": "PERSON", "event-election": "EVENT",
        "event-disaster": "EVENT", "event-protest/demonstration": "EVENT",
        "other-chemicalthing": "SKIP", "other-astronomything": "SKIP",
        "other-currency": "SKIP", "other-disease": "SKIP",
        "other-educationaldegree": "SKIP", "other-god": "PERSON",
        "other-language": "SKIP", "other-livingthing": "SKIP",
        "other-medical": "SKIP", "other-religion": "SKIP",
        "product-car": "SKIP", "product-food": "SKIP",
        "product-instrument": "SKIP", "product-ship": "SKIP",
        "product-weapon": "SKIP", "product-train": "SKIP",
    },
    "GoEmotions": {
        "neutral": "SKIP", "admiration": "EMOTION", "approval": "EMOTION",
        "gratitude": "EMOTION", "annoyance": "EMOTION", "amusement": "EMOTION",
        "curiosity": "EMOTION", "love": "EMOTION", "disapproval": "EMOTION",
        "optimism": "EMOTION", "anger": "EMOTION", "joy": "EMOTION",
        "confusion": "EMOTION", "sadness": "EMOTION", "disappointment": "EMOTION",
        "realization": "EMOTION", "caring": "EMOTION", "surprise": "EMOTION",
        "excitement": "EMOTION", "disgust": "EMOTION", "desire": "EMOTION",
        "fear": "EMOTION", "remorse": "EMOTION", "embarrassment": "EMOTION",
        "nervousness": "EMOTION", "relief": "EMOTION", "pride": "EMOTION",
        "grief": "EMOTION",
    },
    "Emotions Dataset": {
        "happiness": "EMOTION", "sadness": "EMOTION", "neutral": "SKIP",
        "anger": "EMOTION", "love": "EMOTION", "fear": "EMOTION",
        "disgust": "EMOTION", "confusion": "EMOTION", "surprise": "EMOTION",
        "shame": "EMOTION", "guilt": "EMOTION", "sarcasm": "SKIP",
        "desire": "EMOTION",
    },
    "MASSIVE (en-US, slots)": {
        "date": "DATE", "place_name": "PLACE", "event_name": "EVENT",
        "person": "PERSON", "time": "DATE", "media_type": "SKIP",
        "business_name": "ORG", "weather_descriptor": "SKIP",
        "transport_type": "SKIP", "food_type": "SKIP",
        "relation": "RELATIONSHIP_REF", "timeofday": "DATE",
        "artist_name": "PERSON", "definition_word": "SKIP",
        "device_type": "SKIP", "currency_name": "SKIP", "list_name": "SKIP",
        "house_place": "PLACE", "news_topic": "SKIP", "music_genre": "SKIP",
        "business_type": "SKIP", "player_setting": "SKIP",
        "song_name": "SKIP", "radio_name": "ORG", "order_type": "SKIP",
        "color_type": "SKIP", "game_name": "SKIP",
        "general_frequency": "DATE", "audiobook_name": "SKIP",
        "podcast_descriptor": "SKIP", "time_zone": "SKIP",
        "music_descriptor": "SKIP", "transport_agency": "ORG",
        "email_address": "SKIP", "cooking_type": "SKIP", "alarm_type": "SKIP",
    },
    "MultiWOZ v2.2": {
        "restaurant-food": "SKIP", "restaurant-pricerange": "SKIP",
        "restaurant-area": "PLACE", "restaurant-name": "ORG",
        "hotel-name": "ORG", "train-destination": "PLACE",
        "train-departure": "PLACE", "train-day": "DATE",
        "hotel-area": "PLACE", "hotel-pricerange": "SKIP",
        "attraction-type": "SKIP", "train-leaveat": "DATE",
        "train-arriveby": "DATE", "attraction-name": "PLACE",
        "hotel-stars": "SKIP", "attraction-area": "PLACE",
        "hotel-parking": "SKIP", "hotel-internet": "SKIP",
        "restaurant-booktime": "DATE", "hotel-type": "SKIP",
        "restaurant-bookday": "DATE", "restaurant-bookpeople": "SKIP",
        "hotel-bookstay": "DATE", "train-bookpeople": "SKIP",
        "hotel-bookday": "DATE", "hotel-bookpeople": "SKIP",
        "taxi-destination": "PLACE", "taxi-departure": "PLACE",
        "taxi-leaveat": "DATE", "taxi-arriveby": "DATE",
        "hospital-department": "SKIP", "bus-departure": "PLACE",
        "bus-destination": "PLACE", "bus-leaveat": "DATE",
    },
    "Schema-Guided Dialogue": {
        "date": "DATE", "time": "DATE", "city": "PLACE",
        "location": "PLACE", "event_name": "EVENT", "category": "SKIP",
        "address": "PLACE", "number_of_seats": "SKIP",
        "restaurant_name": "ORG", "cuisine": "SKIP", "movie_name": "SKIP",
        "destination": "PLACE", "origin": "PLACE", "price_range": "SKIP",
        "show_time": "DATE", "hotel_name": "ORG",
        "departure_date": "DATE", "phone_number": "SKIP",
        "number_of_tickets": "SKIP", "check_in_date": "DATE",
        "event_type": "SKIP", "check_out_date": "DATE",
        "theater_name": "ORG", "genre": "SKIP", "number_of_rooms": "SKIP",
        "return_date": "DATE", "artist": "PERSON",
        "event_location": "PLACE", "number_of_adults": "SKIP",
        "song_name": "SKIP", "has_live_music": "SKIP",
        "seating_class": "SKIP", "temperature": "SKIP",
        "therapist_name": "PERSON", "account_type": "SKIP",
        "balance": "SKIP", "transfer_amount": "SKIP",
        "dentist_name": "PERSON", "doctor_name": "PERSON",
        "appointment_date": "DATE", "appointment_time": "DATE",
        "property_name": "ORG", "area": "PLACE", "furnished": "SKIP",
        "ride_type": "SKIP", "shared_ride": "SKIP", "ride_fare": "SKIP",
        "airlines": "ORG", "album": "SKIP", "star_rating": "SKIP",
        "pets_allowed": "SKIP",
    },
    "XED (English)": {
        "anger": "EMOTION", "anticipation": "EMOTION", "disgust": "EMOTION",
        "fear": "EMOTION", "joy": "EMOTION", "sadness": "EMOTION",
        "surprise": "EMOTION", "trust": "EMOTION",
    },
    # BANKING77: intent classification labels, NOT entity spans.
    # Useful as Task 9 synthetic generation seed, not NER training data.
    "BANKING77": {k: "SKIP" for k in [
        "card_payment_fee_charged", "direct_debit_payment_not_recognised",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "wrong_amount_of_cash_received", "cash_withdrawal_charge",
        "transaction_charged_twice", "declined_cash_withdrawal",
        "transfer_fee_charged", "transfer_not_received_by_recipient",
        "balance_not_updated_after_bank_transfer", "request_refund",
        "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
        "extra_charge_on_statement", "wrong_exchange_rate_for_cash_withdrawal",
        "Refund_not_showing_up", "reverted_card_payment?",
        "cash_withdrawal_not_recognised", "pending_card_payment",
        "activate_my_card", "cancel_transfer", "beneficiary_not_allowed",
        "card_arrival", "declined_card_payment", "pending_top_up",
        "pending_transfer", "top_up_reverted", "top_up_failed",
        "pending_cash_withdrawal", "card_linking", "country_support",
        "edit_personal_details", "order_physical_card", "card_not_working",
        "virtual_card_not_working", "contactless_not_working",
        "supported_cards_and_currencies", "getting_spare_card",
        "card_about_to_expire", "exchange_via_app", "age_limit",
        "pin_blocked", "receiving_money", "top_up_by_card_charge",
        "getting_virtual_card", "apple_pay_or_google_pay",
        "lost_or_stolen_card", "top_up_limits", "change_pin",
        "exchange_rate", "disposable_card_limits", "fiat_currency_support",
        "why_verify_identity", "lost_or_stolen_phone", "automatic_top_up",
        "visa_or_mastercard", "topping_up_by_card",
        "top_up_by_bank_transfer_charge", "atm_support",
        "unable_to_verify_identity", "verify_my_identity", "verify_top_up",
        "verify_source_of_funds", "get_disposable_virtual_card",
        "passcode_forgotten", "terminate_account", "card_swallowed",
        "failed_transfer", "top_up_by_cash_or_cheque", "get_physical_card",
        "transfer_into_account", "transfer_timing", "card_delivery_estimate",
        "exchange_charge",
    ]},
}

# CrossNER: map integers to names, then names to Oneiron
CROSSNER_TO_ONEIRON = {
    "album": "SKIP", "algorithm": "SKIP", "astronomicalobject": "SKIP",
    "award": "SKIP", "band": "ORG", "book": "SKIP", "country": "PLACE",
    "election": "EVENT", "event": "EVENT", "location": "PLACE",
    "misc": "SKIP", "musicalartist": "PERSON", "organisation": "ORG",
    "person": "PERSON", "politicalparty": "ORG", "politician": "PERSON",
    "scientist": "PERSON", "university": "ORG", "writer": "PERSON",
    "unknown-person": "PERSON", "unknown-organisation": "ORG",
    "unknown-location": "PLACE", "unknown-misc": "SKIP",
    "unknown-politician": "PERSON", "unknown-award": "SKIP",
    "unknown-election": "EVENT", "unknown-politicalparty": "ORG",
    "unknown-scientist": "PERSON", "unknown-band": "ORG",
    "unknown-writer": "PERSON", "unknown-book": "SKIP",
    "unknown-university": "ORG", "unknown-album": "SKIP",
    "unknown-event": "EVENT", "unknown-country": "PLACE",
    "unknown-astronomicalobject": "SKIP", "unknown-musicalartist": "PERSON",
    "unknown-algorithm": "SKIP",
}


def remap_dataset(data: dict, mapping: dict) -> dict:
    type_counts = {}
    for entry in data.get("top_30_types", []):
        raw_type, count = entry[0], entry[1]
        type_counts[raw_type] = count

    # Also gather from oneiron_coverage top_sources
    for ot, info in data.get("oneiron_coverage", {}).items():
        for src in info.get("top_sources", []):
            raw_type, count = src[0], src[1]
            if raw_type not in type_counts:
                type_counts[raw_type] = count

    oneiron_map = defaultdict(lambda: {"count": 0, "top_types": Counter()})
    for raw_type, count in type_counts.items():
        target = mapping.get(raw_type, "SKIP")
        if target != "SKIP":
            oneiron_map[target]["count"] += count
            oneiron_map[target]["top_types"][raw_type] += count
        else:
            oneiron_map["SKIP"]["count"] += count

    new_coverage = {}
    for ot in ONEIRON_TYPES:
        info = oneiron_map.get(ot, {"count": 0, "top_types": Counter()})
        new_coverage[ot] = {
            "count": info["count"],
            "top_sources": info["top_types"].most_common(5),
        }
    new_coverage["UNMAPPED"] = {"count": oneiron_map.get("SKIP", {"count": 0})["count"]}

    data["oneiron_coverage"] = new_coverage
    return data


def remap_crossner(data: dict) -> dict:
    type_counts = {}
    for entry in data.get("top_30_types", []):
        raw, count = str(entry[0]), entry[1]
        if raw in CROSSNER_LABELS:
            label = CROSSNER_LABELS[raw]
            clean = label.replace("B-", "").replace("I-", "")
        else:
            clean = raw
            while clean.startswith("unknown-"):
                clean = clean[len("unknown-"):]
            if clean not in CROSSNER_TO_ONEIRON:
                clean = f"unknown-{clean}"
        type_counts[clean] = type_counts.get(clean, 0) + count

    # Rebuild top_30_types with names
    data["top_30_types"] = sorted(type_counts.items(), key=lambda x: -x[1])[:30]

    oneiron_map = defaultdict(lambda: {"count": 0, "top_types": Counter()})
    for clean_type, count in type_counts.items():
        target = CROSSNER_TO_ONEIRON.get(clean_type, "SKIP")
        if target != "SKIP":
            oneiron_map[target]["count"] += count
            oneiron_map[target]["top_types"][clean_type] += count
        else:
            oneiron_map["SKIP"]["count"] += count

    new_coverage = {}
    for ot in ONEIRON_TYPES:
        info = oneiron_map.get(ot, {"count": 0, "top_types": Counter()})
        new_coverage[ot] = {
            "count": info["count"],
            "top_sources": info["top_types"].most_common(5),
        }
    new_coverage["UNMAPPED"] = {"count": oneiron_map.get("SKIP", {"count": 0})["count"]}

    data["oneiron_coverage"] = new_coverage
    data["unique_types"] = len(type_counts)
    return data


def format_report(analyses):
    lines = [
        "# Green Dataset Analysis — Oneiron Type Coverage",
        "",
        "> Generated: 2026-02-16",
        "> Datasets analyzed: {}".format(len(analyses)),
        "> Type mappings: Verified by Sonnet",
        "",
        "---",
        "",
        "## Summary Table",
        "",
    ]

    header = "| Dataset | Records | Unique Types |"
    for ot in ONEIRON_TYPES:
        header += f" {ot} |"
    lines.append(header)

    sep = "|---|---|---|"
    for _ in ONEIRON_TYPES:
        sep += "---|"
    lines.append(sep)

    for a in analyses:
        row = f"| {a['dataset']} | {a['total_entities']:,} | {a['unique_types']:,} |"
        for ot in ONEIRON_TYPES:
            c = a["oneiron_coverage"].get(ot, {}).get("count", 0)
            row += f" {c:,} |" if c else " — |"
        lines.append(row)

    row = "| **TOTAL** | | |"
    for ot in ONEIRON_TYPES:
        total = sum(a["oneiron_coverage"].get(ot, {}).get("count", 0) for a in analyses)
        row += f" **{total:,}** |"
    lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    for a in analyses:
        lines.append(f"## {a['dataset']}")
        lines.append("")
        lines.append(f"- **Total entities**: {a['total_entities']:,}")
        lines.append(f"- **Unique types**: {a['unique_types']:,}")
        lines.append("")

        lines.append("### Top 30 Entity Types")
        lines.append("| Count | Type |")
        lines.append("|------:|------|")
        for entry in a["top_30_types"]:
            typ, cnt = entry[0], entry[1]
            lines.append(f"| {cnt:,} | {typ} |")
        lines.append("")

        lines.append("### Oneiron Type Coverage")
        lines.append("| Oneiron Type | Count | Top Sources |")
        lines.append("|---|---|---|")
        for ot in ONEIRON_TYPES:
            info = a["oneiron_coverage"].get(ot, {"count": 0, "top_sources": []})
            sources = ", ".join(f"{t} ({c:,})" for t, c in info.get("top_sources", [])[:3])
            c = info["count"]
            lines.append(f"| {ot} | {c:,} | {sources} |")
        unmapped = a["oneiron_coverage"].get("UNMAPPED", {}).get("count", 0)
        lines.append(f"| SKIP | {unmapped:,} | |")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        name = data["dataset"]

        if name == "CrossNER":
            data = remap_crossner(data)
            print(f"  Remapped {name} (integer labels → names → Oneiron)")
        elif name in VERIFIED_MAPPINGS:
            data = remap_dataset(data, VERIFIED_MAPPINGS[name])
            print(f"  Remapped {name}")
        else:
            print(f"  Unchanged {name} (no verified mapping)")

        # Save updated result
        with open(f, "w") as fp:
            json.dump(data, fp, indent=2, default=str)

        results.append(data)

    results.sort(key=lambda r: (-r["total_entities"], r["dataset"]))

    md_path = REPORT_DIR / "GREEN-DATASET-ANALYSIS.md"
    report = format_report(results)

    # Append unlabeled datasets section
    note_lines = ["\n## Unlabeled Datasets (Need LLM Pass)\n"]
    note_lines.append("| Dataset | License | Notes |")
    note_lines.append("|---------|---------|-------|")
    for r in results:
        if r.get("note"):
            note_lines.append(f"| {r['dataset']} | — | {r['note']} |")
    if len(note_lines) > 3:
        report += "\n".join(note_lines) + "\n"

    with open(md_path, "w") as fp:
        fp.write(report)
    print(f"\nReport written to {md_path}")

    # Print summary
    print("\n=== Corrected Totals ===")
    for ot in ONEIRON_TYPES:
        total = sum(r["oneiron_coverage"].get(ot, {}).get("count", 0) for r in results)
        if total:
            print(f"  {ot:20s}: {total:>10,}")


if __name__ == "__main__":
    main()
