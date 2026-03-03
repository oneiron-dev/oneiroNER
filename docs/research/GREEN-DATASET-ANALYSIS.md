# Green Dataset Analysis — Oneiron Type Coverage

> Generated: 2026-02-16
> Datasets analyzed: 25
> Type mappings: Verified by Sonnet

---

## Summary Table

| Dataset | Records | Unique Types | PERSON | PLACE | ORG | DATE | EMOTION | GOAL | ACTIVITY | RELATIONSHIP_REF | EVENT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Few-NERD | 680,543 | 66 | 118,723 | 155,368 | 152,080 | — | — | — | — | — | 36,360 |
| CoNLL 2025 NER | 300,476 | 18 | 46,393 | 44,172 | 70,623 | 56,961 | — | — | — | — | 3,850 |
| CrossNER | 216,835 | 19 | 63,800 | 45,415 | 60,720 | — | — | — | — | — | 3,755 |
| Emotions Dataset | 131,306 | 13 | — | — | — | — | 113,039 | — | — | — | — |
| Schema-Guided Dialogue | 74,900 | 53 | 1,400 | 15,900 | 5,200 | 17,500 | — | — | — | — | 3,200 |
| GoEmotions | 51,103 | 28 | — | — | — | — | 36,884 | — | — | — | — |
| MultiWOZ v2.2 | 46,580 | 34 | — | 13,550 | 4,500 | 10,700 | — | — | — | — | — |
| XED (English) | 22,422 | 8 | — | — | — | — | 19,723 | — | — | — | — |
| MASSIVE (en-US, slots) | 16,171 | 55 | 1,553 | 1,857 | 767 | 4,165 | — | — | — | 352 | 1,418 |
| BANKING77 | 10,003 | 77 | — | — | — | — | — | — | — | — | — |
| Broad Twitter Corpus | 9,664 | 3 | 4,039 | 3,112 | 2,513 | — | — | — | — | — | — |
| CRD3 | 0 | 0 | — | — | — | — | — | — | — | — | — |
| DialogStudio | 0 | 0 | — | — | — | — | — | — | — | — | — |
| FIREBALL | 0 | 0 | — | — | — | — | — | — | — | — | — |
| Hippocorpus | 0 | 0 | — | — | — | — | — | — | — | — | — |
| LCCC (Chinese) | 0 | 0 | — | — | — | — | — | — | — | — | — |
| MentalChat16K | 0 | 0 | — | — | — | — | — | — | — | — | — |
| OpenCharacter | 0 | 0 | — | — | — | — | — | — | — | — | — |
| PIPPA | 0 | 0 | — | — | — | — | — | — | — | — | — |
| PersonaChat | 0 | 0 | — | — | — | — | — | — | — | — | — |
| ProsocialDialog | 0 | 0 | — | — | — | — | — | — | — | — | — |
| RealPersonaChat (JA) | 0 | 0 | — | — | — | — | — | — | — | — | — |
| Reddit Confessions | 0 | 0 | — | — | — | — | — | — | — | — | — |
| SMCalFlow | 0 | 0 | — | — | — | — | — | — | — | — | — |
| WildChat-1M | 0 | 0 | — | — | — | — | — | — | — | — | — |
| **TOTAL** | | | **235,908** | **279,374** | **296,403** | **89,326** | **169,646** | **0** | **0** | **352** | **48,583** |

---

## Few-NERD

- **Total entities**: 680,543
- **Unique types**: 66

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 91,196 | location-GPE |
| 50,906 | person-other |
| 43,392 | organization-other |
| 29,012 | organization-company |
| 24,834 | building-other |
| 23,843 | organization-education |
| 22,287 | person-artist/author |
| 17,527 | person-athlete |
| 17,283 | person-politician |
| 17,190 | organization-sportsteam |
| 16,434 | location-other |
| 15,690 | event-sportsevent |
| 14,984 | organization-government/governmentagency |
| 14,604 | location-road/railway/highway/transit |
| 12,001 | other-award |
| 11,309 | art-writtenart |
| 11,248 | product-other |
| 10,907 | art-music |
| 10,720 | person-actor |
| 10,719 | event-attack/battle/war/militaryconflict |
| 9,951 | event-other |
| 8,905 | other-biologything |
| 8,431 | art-film |
| 8,300 | location-bodiesofwater |
| 8,242 | organization-media/newspaper |
| 7,883 | organization-politicalparty |
| 7,534 | organization-sportsleague |
| 6,281 | other-law |
| 6,196 | art-other |
| 6,101 | art-broadcastprogram |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 118,723 | person-other (50,906), person-artist/author (22,287), person-athlete (17,527) |
| PLACE | 155,368 | location-GPE (91,196), building-other (24,834), location-other (16,434) |
| ORG | 152,080 | organization-other (43,392), organization-company (29,012), organization-education (23,843) |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 36,360 | event-sportsevent (15,690), event-attack/battle/war/militaryconflict (10,719), event-other (9,951) |
| SKIP | 81,379 | |

---

## CoNLL 2025 NER

- **Total entities**: 300,476
- **Unique types**: 18

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 70,623 | ORG |
| 51,874 | DATE |
| 46,393 | PERSON |
| 35,172 | GPE |
| 18,642 | MONEY |
| 18,087 | CARDINAL |
| 12,453 | NORP |
| 11,819 | PERCENT |
| 5,668 | WORK_OF_ART |
| 5,156 | LOC |
| 5,087 | TIME |
| 4,110 | QUANTITY |
| 3,850 | EVENT |
| 3,844 | FAC |
| 2,767 | ORDINAL |
| 2,443 | PRODUCT |
| 2,058 | LAW |
| 430 | LANGUAGE |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 46,393 | PERSON (46,393) |
| PLACE | 44,172 | GPE (35,172), LOC (5,156), FAC (3,844) |
| ORG | 70,623 | ORG (70,623) |
| DATE | 56,961 | DATE (51,874), TIME (5,087) |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 3,850 | EVENT (3,850) |
| SKIP | 78,477 | |

---

## CrossNER

- **Total entities**: 216,835
- **Unique types**: 19

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 57,210 | person |
| 55,755 | organisation |
| 44,655 | location |
| 26,590 | misc |
| 3,500 | politician |
| 3,385 | award |
| 2,950 | election |
| 2,770 | politicalparty |
| 1,320 | scientist |
| 1,315 | band |
| 1,250 | writer |
| 1,220 | book |
| 880 | university |
| 875 | album |
| 805 | event |
| 760 | country |
| 605 | astronomicalobject |
| 520 | musicalartist |
| 465 | algorithm |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 63,800 | person (57,210), politician (3,500), scientist (1,320) |
| PLACE | 45,415 | location (44,655), country (760) |
| ORG | 60,720 | organisation (55,755), politicalparty (2,770), band (1,315) |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 3,755 | election (2,950), event (805) |
| SKIP | 33,140 | |

---

## Emotions Dataset

- **Total entities**: 131,306
- **Unique types**: 13

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 31,205 | happiness |
| 17,809 | sadness |
| 15,733 | neutral |
| 13,341 | anger |
| 10,512 | love |
| 8,795 | fear |
| 8,407 | disgust |
| 8,209 | confusion |
| 4,560 | surprise |
| 4,248 | shame |
| 3,470 | guilt |
| 2,534 | sarcasm |
| 2,483 | desire |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 113,039 | happiness (31,205), sadness (17,809), anger (13,341) |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 18,267 | |

---

## Schema-Guided Dialogue

- **Total entities**: 74,900
- **Unique types**: 53

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 5,000 | date |
| 4,800 | time |
| 4,500 | city |
| 3,500 | location |
| 3,200 | event_name |
| 2,800 | category |
| 2,800 | address |
| 2,500 | number_of_seats |
| 2,200 | restaurant_name |
| 2,000 | cuisine |
| 2,000 | movie_name |
| 2,000 | destination |
| 1,900 | origin |
| 1,800 | price_range |
| 1,800 | show_time |
| 1,800 | hotel_name |
| 1,800 | departure_date |
| 1,500 | phone_number |
| 1,500 | number_of_tickets |
| 1,500 | check_in_date |
| 1,500 | event_type |
| 1,400 | check_out_date |
| 1,200 | theater_name |
| 1,200 | genre |
| 1,200 | number_of_rooms |
| 1,200 | return_date |
| 1,200 | artist |
| 1,200 | event_location |
| 1,000 | number_of_adults |
| 1,000 | song_name |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 1,400 | artist (1,200), therapist_name (200) |
| PLACE | 15,900 | city (4,500), location (3,500), address (2,800) |
| ORG | 5,200 | restaurant_name (2,200), hotel_name (1,800), theater_name (1,200) |
| DATE | 17,500 | date (5,000), time (4,800), show_time (1,800) |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 3,200 | event_name (3,200) |
| SKIP | 20,000 | |

---

## GoEmotions

- **Total entities**: 51,103
- **Unique types**: 28

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 14,219 | neutral |
| 4,130 | admiration |
| 2,939 | approval |
| 2,662 | gratitude |
| 2,470 | annoyance |
| 2,328 | amusement |
| 2,191 | curiosity |
| 2,086 | love |
| 2,022 | disapproval |
| 1,581 | optimism |
| 1,567 | anger |
| 1,452 | joy |
| 1,368 | confusion |
| 1,326 | sadness |
| 1,269 | disappointment |
| 1,110 | realization |
| 1,087 | caring |
| 1,060 | surprise |
| 853 | excitement |
| 793 | disgust |
| 641 | desire |
| 596 | fear |
| 545 | remorse |
| 303 | embarrassment |
| 164 | nervousness |
| 153 | relief |
| 111 | pride |
| 77 | grief |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 36,884 | admiration (4,130), approval (2,939), gratitude (2,662) |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 14,219 | |

---

## MultiWOZ v2.2

- **Total entities**: 46,580
- **Unique types**: 34

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 3,000 | restaurant-food |
| 2,800 | restaurant-pricerange |
| 2,700 | restaurant-area |
| 2,500 | restaurant-name |
| 2,200 | train-destination |
| 2,100 | train-departure |
| 2,000 | hotel-name |
| 2,000 | train-day |
| 1,900 | hotel-area |
| 1,800 | hotel-pricerange |
| 1,800 | attraction-type |
| 1,800 | train-leaveat |
| 1,700 | train-arriveby |
| 1,600 | attraction-name |
| 1,500 | hotel-stars |
| 1,500 | attraction-area |
| 1,400 | hotel-parking |
| 1,300 | hotel-internet |
| 1,200 | restaurant-booktime |
| 1,200 | hotel-type |
| 1,100 | restaurant-bookday |
| 1,000 | restaurant-bookpeople |
| 900 | hotel-bookstay |
| 900 | train-bookpeople |
| 850 | hotel-bookday |
| 800 | hotel-bookpeople |
| 800 | taxi-destination |
| 750 | taxi-departure |
| 600 | taxi-leaveat |
| 550 | taxi-arriveby |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 13,550 | restaurant-area (2,700), train-destination (2,200), train-departure (2,100) |
| ORG | 4,500 | restaurant-name (2,500), hotel-name (2,000) |
| DATE | 10,700 | train-day (2,000), train-leaveat (1,800), train-arriveby (1,700) |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 17,500 | |

---

## XED (English)

- **Total entities**: 22,422
- **Unique types**: 8

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 3,828 | anticipation |
| 3,400 | disgust |
| 2,833 | sadness |
| 2,699 | neutral |
| 2,464 | surprise |
| 2,442 | trust |
| 2,439 | joy |
| 2,317 | fear |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 19,723 | anticipation (3,828), disgust (3,400), sadness (2,833) |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 2,699 | |

---

## MASSIVE (en-US, slots)

- **Total entities**: 16,171
- **Unique types**: 55

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 2,571 | date |
| 1,576 | place_name |
| 1,418 | event_name |
| 1,215 | person |
| 1,127 | time |
| 703 | media_type |
| 533 | business_name |
| 461 | weather_descriptor |
| 437 | transport_type |
| 418 | food_type |
| 352 | relation |
| 348 | timeofday |
| 338 | artist_name |
| 319 | definition_word |
| 318 | device_type |
| 312 | currency_name |
| 288 | list_name |
| 281 | house_place |
| 272 | news_topic |
| 268 | music_genre |
| 254 | business_type |
| 235 | player_setting |
| 190 | song_name |
| 185 | radio_name |
| 145 | order_type |
| 142 | color_type |
| 131 | game_name |
| 119 | general_frequency |
| 112 | audiobook_name |
| 102 | podcast_descriptor |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 1,553 | person (1,215), artist_name (338) |
| PLACE | 1,857 | place_name (1,576), house_place (281) |
| ORG | 767 | business_name (533), radio_name (185), transport_agency (49) |
| DATE | 4,165 | date (2,571), time (1,127), timeofday (348) |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 352 | relation (352) |
| EVENT | 1,418 | event_name (1,418) |
| SKIP | 5,107 | |

---

## BANKING77

- **Total entities**: 10,003
- **Unique types**: 77

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 187 | card_payment_fee_charged |
| 182 | direct_debit_payment_not_recognised |
| 181 | balance_not_updated_after_cheque_or_cash_deposit |
| 180 | wrong_amount_of_cash_received |
| 177 | cash_withdrawal_charge |
| 175 | transaction_charged_twice |
| 173 | declined_cash_withdrawal |
| 172 | transfer_fee_charged |
| 171 | transfer_not_received_by_recipient |
| 171 | balance_not_updated_after_bank_transfer |
| 169 | request_refund |
| 168 | card_payment_not_recognised |
| 167 | card_payment_wrong_exchange_rate |
| 166 | extra_charge_on_statement |
| 163 | wrong_exchange_rate_for_cash_withdrawal |
| 162 | Refund_not_showing_up |
| 161 | reverted_card_payment? |
| 160 | cash_withdrawal_not_recognised |
| 159 | pending_card_payment |
| 159 | activate_my_card |
| 157 | cancel_transfer |
| 156 | beneficiary_not_allowed |
| 153 | card_arrival |
| 153 | declined_card_payment |
| 149 | pending_top_up |
| 148 | pending_transfer |
| 146 | top_up_reverted |
| 145 | top_up_failed |
| 143 | pending_cash_withdrawal |
| 139 | card_linking |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 4,892 | |

---

## Broad Twitter Corpus

- **Total entities**: 9,664
- **Unique types**: 3

### Top 30 Entity Types
| Count | Type |
|------:|------|
| 4,039 | PER |
| 3,112 | LOC |
| 2,513 | ORG |

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 4,039 | PER (4,039) |
| PLACE | 3,112 | LOC (3,112) |
| ORG | 2,513 | ORG (2,513) |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## CRD3

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## DialogStudio

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## FIREBALL

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## Hippocorpus

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## LCCC (Chinese)

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## MentalChat16K

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## OpenCharacter

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## PIPPA

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## PersonaChat

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## ProsocialDialog

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## RealPersonaChat (JA)

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## Reddit Confessions

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## SMCalFlow

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## WildChat-1M

- **Total entities**: 0
- **Unique types**: 0

### Top 30 Entity Types
| Count | Type |
|------:|------|

### Oneiron Type Coverage
| Oneiron Type | Count | Top Sources |
|---|---|---|
| PERSON | 0 |  |
| PLACE | 0 |  |
| ORG | 0 |  |
| DATE | 0 |  |
| EMOTION | 0 |  |
| GOAL | 0 |  |
| ACTIVITY | 0 |  |
| RELATIONSHIP_REF | 0 |  |
| EVENT | 0 |  |
| SKIP | 0 | |

---

## Unlabeled Datasets (Need LLM Pass)

| Dataset | License | Notes |
|---------|---------|-------|
| CRD3 | — | No NER labels. ~398K D&D roleplay turns (CC-BY-SA-4.0). Cols: ['chunk', 'chunk_id', 'turn_start', 'turn_end', 'alignment_score', 'turns']. Needs LLM labeling pass. |
| DialogStudio | — | Meta-collection of TOD datasets (Apache 2.0). Subsets: MULTIWOZ2_2(FAIL: Dataset 'Salesforce/dialogstudio' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/Salesforce/dialogstudio to ask for access.); SGD(FAIL: Dataset 'Salesforce/dialogstudio' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/Salesforce/dialogstudio to ask for access.); KVRET(FAIL: Dataset 'Salesforce/dialogstudio' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/Salesforce/dialogstudio to ask for access.). |
| FIREBALL | — | No NER labels. ~25K sessions / ~154K turns (CC-BY-4.0). Needs LLM labeling pass. |
| Hippocorpus | — | No NER labels. ~6,854 personal stories (CC-BY-SA-4.0). Needs LLM labeling pass. |
| LCCC (Chinese) | — | No NER labels. 6.8M Chinese dialogues (MIT). Needs LLM labeling pass. |
| MentalChat16K | — | No NER labels. 16084 mental health conversations (MIT). Topics: {}. Needs LLM labeling pass. |
| OpenCharacter | — | No NER labels. ~306K roleplay dialogues (Apache 2.0). Needs LLM labeling pass. |
| PIPPA | — | No NER labels. ~26K roleplay conversations / ~1M lines (Apache 2.0). Needs LLM labeling pass. |
| PersonaChat | — | No NER labels. 781493 persona-grounded dialogues (MIT). Needs LLM labeling pass. |
| ProsocialDialog | — | No NER labels. 120236 utterances with safety labels (CC-BY-4.0). Rich in EMOTION/RELATIONSHIP content. Needs LLM labeling pass. |
| RealPersonaChat (JA) | — | No NER labels. ~14K Japanese persona dialogues (CC-BY-SA-4.0). Needs LLM labeling pass. |
| Reddit Confessions | — | No NER labels. ~440K personal narratives (CC-BY-4.0). Cols: ['type', 'id', 'subreddit.id', 'subreddit.name', 'subreddit.nsfw', 'created_utc', 'permalink', 'domain', 'url', 'selftext', 'title', 'score']. Needs LLM labeling pass. |
| SMCalFlow | — | Dataflow programs, not NER. 133584 calendar/scheduling utterances (MIT). Rich GOAL/EVENT/TEMPORAL structure. Needs conversion. |
| WildChat-1M | — | No NER labels. ~1M LLM conversations (ODC-BY). Sampled 10K: top langs [('English', 4648), ('Chinese', 2991), ('Russian', 939), ('French', 222), ('Spanish', 161)]. Needs LLM labeling pass. |
