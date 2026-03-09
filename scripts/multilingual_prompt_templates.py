#!/usr/bin/env python3
"""Prompt templates for multilingual NER conversation generation.

Usage:
    from multilingual_prompt_templates import build_prompt
    prompt = build_prompt("da", count=50, mode="batch", batch_num=1)
"""

ENTITY_TYPES = """PERSON, RELATIONSHIP_REF/Family, RELATIONSHIP_REF/Friend, RELATIONSHIP_REF/Romantic,
RELATIONSHIP_REF/Professional, RELATIONSHIP_REF/Acquaintance, PLACE, ORG,
DATE/Day, DATE/Week, DATE/Month, DATE/Year, DATE/Relative, DATE/Holiday,
EVENT, EVENT/Life, EVENT/General, EMOTION, GOAL, ACTIVITY"""

LANG_CONFIG = {
    "de": {
        "name": "German", "script": "latin",
        "chars": "ä, ö, ü, ß (NEVER ae, oe, ue, ss substitutes — use PROPER Unicode umlauts)",
        "cultural": "Feierabend, Stammtisch, Schützenfest, Karneval, Weihnachtsmarkt, Brotzeit, Abendbrot",
        "places": "Berlin, München, Hamburg, Köln, Frankfurt, Stuttgart, Düsseldorf, Leipzig",
        "names": "Lukas, Maximilian, Felix, Jonas, Lena, Sophie, Laura, Anna, Tobias, Moritz",
        "family": "meine Mutter, mein Vater, mein Bruder, meine Schwester, Oma, Opa, Schwiegermutter",
        "notes": "CRITICAL: Use ä ö ü ß — NEVER ASCII substitutes. This was the original issue with GPT-5.4 German.",
    },
    "da": {
        "name": "Danish", "script": "latin",
        "chars": "æ, ø, å (NEVER ae, oe, aa)",
        "cultural": "hygge, kolonihave, folkeskole, gymnasium, SU, julefrokost, Roskilde Festival, fællesspisning",
        "places": "København, Aarhus, Aalborg, Odense, Roskilde, Bornholm, Nyhavn, Tivoli",
        "names": "Lars, Morten, Anders, Søren, Mette, Line, Camilla, Sofie, Kasper, Mikkel",
        "family": "mor, far, bror, søster, mormor, morfar, farmor, farfar, svigermor",
    },
    "no": {
        "name": "Norwegian Bokmål", "script": "latin",
        "chars": "æ, ø, å (NEVER ae, oe, aa)",
        "cultural": "hytte, bunad, matpakke, barnehage, dugnad, russ, Vinmonopolet",
        "places": "Oslo, Bergen, Tromsø, Trondheim, Stavanger, Lofoten, Nordkapp",
        "names": "Erik, Olav, Sigurd, Magnus, Ingrid, Sigrid, Astrid, Nora, Sander, Emilie",
        "family": "mamma, pappa, broren min, bestemor, svigermor",
    },
    "fi": {
        "name": "Finnish", "script": "latin",
        "chars": "ä, ö (NEVER ae, oe)",
        "cultural": "sauna, mökki, juhannus, vappu, KELA, pikkujoulu, runeberginpäivä",
        "places": "Helsinki, Tampere, Turku, Oulu, Rovaniemi, Lappi, Suomenlinna",
        "names": "Matti, Mikko, Jukka, Antti, Liisa, Sari, Tiina, Päivi, Ville, Lauri",
        "family": "äiti, isä, sisko, veli, mummo, vaari, anoppi",
        "notes": "Finnish has long compound words — annotate them correctly. Use puhekieli where natural.",
    },
    "hr": {
        "name": "Croatian", "script": "latin",
        "chars": "č, ć, đ, š, ž (NEVER c, dj, s, z substitutes)",
        "cultural": "Advent u Zagrebu, ćevapi, kava, fjaka, more, rivijera",
        "places": "Zagreb, Split, Dubrovnik, Rijeka, Zadar, Hvar, Plitvice",
        "names": "Marko, Ivan, Luka, Matej, Ana, Ivana, Petra, Maja, Ante, Tomislav",
        "family": "mama, tata, brat, sestra, baka, djed, stric, teta",
    },
    "hu": {
        "name": "Hungarian", "script": "latin",
        "chars": "á, é, í, ó, ö, ő, ú, ü, ű (NEVER substitute)",
        "cultural": "gulyás, pálinka, Balaton, strand, fröccs, buli, koli, suli",
        "places": "Budapest, Debrecen, Szeged, Pécs, Balaton, Eger, Hévíz",
        "names": "László, István, Gábor, Péter, Katalin, Zsófi, Réka, Ági, Bence, Dávid",
        "family": "anyám, apám, bátyám, húgom, nagymama, nagypapa, anyósom",
        "notes": "Hungarian name order: Family+Given. Agglutinative suffixes OK in surfaces.",
    },
    "sk": {
        "name": "Slovak", "script": "latin",
        "chars": "á, ä, č, ď, é, í, ľ, ĺ, ň, ó, ô, ŕ, š, ť, ú, ý, ž",
        "cultural": "bryndzové halušky, salaš, Tatry, kúpele, kapustnica, Silvester",
        "places": "Bratislava, Košice, Žilina, Banská Bystrica, Tatry, Piešťany",
        "names": "Jozef, Peter, Marek, Tomáš, Mária, Jana, Zuzana, Katarína, Matej, Filip",
        "family": "mama, otec, brat, sestra, babka, dedko, svokra",
    },
    "et": {
        "name": "Estonian", "script": "latin",
        "chars": "ä, ö, ü, õ (NEVER ae, oe, ue — note: õ is unique to Estonian)",
        "cultural": "jaanipäev, laulupidu, saunapäev, maasikakorjamine, talveujumine",
        "places": "Tallinn, Tartu, Pärnu, Saaremaa, Haapsalu, Otepää",
        "names": "Mati, Jüri, Andres, Toomas, Kati, Tiina, Liina, Marika, Raivo, Peeter",
        "family": "ema, isa, vend, õde, vanaema, vanaisa, ämm",
    },
    "lt": {
        "name": "Lithuanian", "script": "latin",
        "chars": "ą, č, ę, ė, į, š, ų, ū, ž (NEVER substitute)",
        "cultural": "cepelinai, krepšinis, Joninės, Kaziuko mugė, Kuršių nerija",
        "places": "Vilnius, Kaunas, Klaipėda, Šiauliai, Palanga, Trakai, Neringa",
        "names": "Jonas, Petras, Vytautas, Mindaugas, Ona, Rūta, Giedrė, Daiva, Lukas, Tomas",
        "family": "mama, tėtis, brolis, sesuo, močiutė, senelis, uošvė",
        "notes": "Lithuanian has grammatical cases — entity surfaces match text form.",
    },
    "lv": {
        "name": "Latvian", "script": "latin",
        "chars": "ā, č, ē, ģ, ī, ķ, ļ, ņ, š, ū, ž (NEVER substitute)",
        "cultural": "Jāņi/Līgo, Dziesmu svētki, pelēkie zirņi, pirts",
        "places": "Rīga, Liepāja, Jūrmala, Daugavpils, Sigulda, Cēsis",
        "names": "Jānis, Andris, Mārtiņš, Kārlis, Līga, Ilze, Baiba, Dace, Edgars, Raivis",
        "family": "mamma, tētis, brālis, māsa, vecāmamma, vectēvs",
    },
    "ca": {
        "name": "Catalan", "script": "latin",
        "chars": "à, è, é, í, ï, ò, ó, ú, ü, ç, l·l (punt volat!)",
        "cultural": "Sant Jordi (roses i llibres), calçotada, castells, La Mercè, caganer",
        "places": "Barcelona, Girona, Tarragona, Lleida, Montserrat, Costa Brava",
        "names": "Jordi, Pere, Marc, Oriol, Montse, Laia, Núria, Marta, Arnau, Pol",
        "family": "la mare, el pare, el germà, la germana, l'àvia, l'avi, la sogra",
        "notes": "IMPORTANT: Catalan uses l·l (ela geminada with punt volat) not l.l or ll",
    },
    "af": {
        "name": "Afrikaans", "script": "latin",
        "chars": "ê, ë, î, ô, û (NEVER substitute)",
        "cultural": "braai, biltong, boerewors, rugby, Tafelberg, potjiekos, lekker",
        "places": "Kaapstad, Johannesburg, Pretoria, Stellenbosch, Durban, Tafelberg",
        "names": "Pieter, Johan, Hennie, Willem, Annemarie, Elna, Marietjie, Riaan, Danie, Kobus",
        "family": "my ma, my pa, my broer, my suster, ouma, oupa, skoonma",
        "notes": "Must sound AFRIKAANS, not Dutch!",
    },
    "tl": {
        "name": "Filipino/Tagalog", "script": "latin",
        "chars": "Standard Latin (no special chars)",
        "cultural": "bayanihan, fiesta, simbang gabi, noche buena, merienda, jeepney, palengke",
        "places": "Maynila, Quezon City, Cebu, Davao, Baguio, Boracay, Palawan",
        "names": "Juan, Jose, Maria, Ate Ging, Kuya Rodel, Tita Celia, Mang Boy, Aling Nena",
        "family": "nanay ko, tatay ko, kuya ko, ate ko, lola, lolo, tita, tito, biyenan ko",
        "notes": "Taglish code-switching is NATURAL and expected.",
    },
    "ms": {
        "name": "Malay", "script": "latin",
        "chars": "Standard Latin",
        "cultural": "pasar malam, roti canai, teh tarik, mamak, kampung, balik kampung, Hari Raya",
        "places": "Kuala Lumpur, Penang, Johor Bahru, Melaka, Langkawi, Kota Kinabalu",
        "names": "Ahmad, Hafiz, Amirul, Faiz, Siti, Nurul, Aina, Syafiq, Danial, Afiqah",
        "family": "mak, ayah, abang, kakak, adik, nenek, atuk, mak mertua",
        "notes": "Use colloquial: 'nak' not 'hendak', 'tak' not 'tidak'. Particles: lah, la, kan.",
    },
    "sw": {
        "name": "Swahili", "script": "latin",
        "chars": "Standard Latin",
        "cultural": "chai, ugali, nyama choma, boda boda, M-Pesa, dala dala, safari",
        "places": "Dar es Salaam, Nairobi, Mombasa, Zanzibar, Arusha, Kilimanjaro",
        "names": "Juma, Hassan, Bakari, Mwangi, Amina, Fatma, Halima, Zawadi, Baraka, Salim",
        "family": "mama yangu, baba yangu, kaka yangu, dada yangu, bibi, babu, mama mkwe",
        "notes": "Include both Kenyan and Tanzanian Swahili variants.",
    },
    "el": {
        "name": "Greek", "script": "greek",
        "chars": "Greek alphabet with accents: ά, έ, ή, ί, ό, ύ, ώ. NEVER transliterate to Latin!",
        "cultural": "φραπέ, καφενείο, ταβέρνα, Πάσχα, αρνί, νησιά, φροντιστήριο, ΚΤΕΛ",
        "places": "Αθήνα, Θεσσαλονίκη, Κρήτη, Σαντορίνη, Μύκονος, Πάτρα",
        "names": "Γιώργος, Νίκος, Δημήτρης, Κώστας, Μαρία, Ελένη, Κατερίνα, Σοφία, Αλέξης, Θάνος",
        "family": "η μάνα μου, ο πατέρας μου, ο αδερφός μου, η αδερφή μου, η γιαγιά, ο παππούς",
    },
    "he": {
        "name": "Hebrew", "script": "hebrew",
        "chars": "Hebrew alphabet א-ת. NO niqqud (vowel points). Use geresh ׳ and gershayim ״.",
        "cultural": "שוק, חומוס, פלאפל, קיבוץ, מילואים, צה״ל, בית קפה",
        "places": "תל אביב, ירושלים, חיפה, באר שבע, אילת, הרצליה",
        "names": "יוסי, אבי, עומר, נועם, דנה, מיכל, שירה, ליאור, איתי, רוני",
        "family": "אמא שלי, אבא שלי, אח שלי, אחות שלי, סבתא, סבא, חמות שלי",
        "notes": "RTL display but LTR Python indexing — offsets work normally.",
    },
    "fa": {
        "name": "Persian/Farsi", "script": "persian",
        "chars": "Arabic-derived + پ, چ, ژ, گ. Use Persian ی (U+06CC) and ک (U+06A9), NOT Arabic ي/ك.",
        "cultural": "نوروز, شب یلدا, چهارشنبه‌سوری, تعارف, چای, بازار, سیزده‌به‌در",
        "places": "تهران, اصفهان, شیراز, مشهد, تبریز, کیش",
        "names": "علی, رضا, حسین, محمد, مریم, سارا, فاطمه, نازنین, امیر, پریسا",
        "family": "مامانم, بابام, داداشم, خواهرم, مادربزرگم, پدربزرگم, مادرشوهرم",
        "notes": "ZWNJ (U+200C) counts as 1 in len(). Colloquial: می‌خوام not می‌خواهم.",
    },
    "bg": {
        "name": "Bulgarian", "script": "cyrillic",
        "chars": "Bulgarian Cyrillic. NEVER mix Latin lookalikes (е≠e, а≠a, о≠o).",
        "cultural": "баница, шопска салата, ракия, именни дни, Черно море",
        "places": "София, Пловдив, Варна, Бургас, Велико Търново, Банско",
        "names": "Георги, Иван, Димитър, Стоян, Мария, Петя, Десислава, Калин, Борис, Надежда",
        "family": "майка ми, баща ми, брат ми, сестра ми, баба ми, дядо ми, свекърва ми",
    },
    "sr": {
        "name": "Serbian", "script": "cyrillic+latin",
        "chars": "Digraphic: Cyrillic (А-Я + Ђ,Љ,Њ,Ћ,Џ) AND Latin (č,ć,đ,š,ž). NEVER mix in one convo.",
        "cultural": "ćevapi, rakija, slava, EXIT festival, Kalemegdan, kafana, ajvar",
        "places": "Београд/Beograd, Нови Сад/Novi Sad, Ниш/Niš",
        "names": "Марко/Marko, Никола/Nikola, Стефан/Stefan, Јована/Jovana, Милица/Milica",
        "family": "мама/mama, тата/tata, брат/brat, сестра/sestra, баба/baba, деда/deda",
        "notes": "Split 50/50 between Cyrillic and Latin script conversations.",
    },
    "bn": {
        "name": "Bengali", "script": "bengali",
        "chars": "Bengali script. Combining matras/hasanta — len() counts codepoints.",
        "cultural": "ঈদ, দুর্গাপূজা, পহেলা বৈশাখ, ইলিশ, আড্ডা, বিকাশ, রিকশা",
        "places": "ঢাকা, কলকাতা, চট্টগ্রাম, সিলেট, রাজশাহী, কক্সবাজার",
        "names": "রহিম, করিম, সুমন, রাজু, প্রিয়া, অনিমা, শ্রাবণী, তানভীর, ফারহান, মিথিলা",
        "family": "আমার মা, আমার বাবা, আমার ভাই, আমার বোন, দাদু, দিদিমা, শাশুড়ি",
        "notes": "Mix Bangladeshi and West Bengali styles.",
    },
    "ta": {
        "name": "Tamil", "script": "tamil",
        "chars": "Tamil script. Combining vowel signs. Use text.find() for offsets.",
        "cultural": "filter kaapi, idli-dosai, kolam, temple, cinema, Pongal",
        "places": "சென்னை, மதுரை, கோயம்புத்தூர், திருச்சி, ஊட்டி, ராமேஸ்வரம்",
        "names": "முருகன், செல்வம், கார்த்திக், சுரேஷ், லட்சுமி, பிரியா, அருண், விஜய்",
        "family": "என் அம்மா, என் அப்பா, என் அண்ணன், என் அக்கா, பாட்டி, தாத்தா",
        "notes": "Include Tanglish code-switching.",
    },
    "te": {
        "name": "Telugu", "script": "telugu",
        "chars": "Telugu script. Combining vowel signs. Use text.find() for offsets.",
        "cultural": "biryani, Charminar, Bathukamma, Bonalu, Tirupati, filter coffee",
        "places": "హైదరాబాద్, విశాఖపట్నం, విజయవాడ, తిరుపతి, వరంగల్",
        "names": "రాజు, సురేష్, వెంకట్, ప్రసాద్, లక్ష్మి, ప్రియ, అనుష, శ్రీనివాస్, రవి, మహేష్",
        "family": "మా అమ్మ, మా నాన్న, మా అన్న, మా అక్క, నాన్నమ్మ, తాతయ్య, అత్తగారు",
        "notes": "Include Tenglish code-switching.",
    },
    "ur": {
        "name": "Urdu", "script": "urdu",
        "chars": "Nastaliq Arabic-derived RTL. Extra: پ,ٹ,ڈ,ڑ,ں,ے,ھ,گ. No diacritics in casual text.",
        "cultural": "چائے, بریانی, کرکٹ, بازار, رمضان, عید",
        "places": "لاہور, کراچی, اسلام آباد, راولپنڈی, فیصل آباد, مری",
        "names": "احمد, عمران, بلال, فاطمہ, عائشہ, حسن, سارہ, علی, زینب, حمزہ",
        "family": "امی, ابو, بھائی, بہن, دادی, دادا, نانی, ساس",
        "notes": "Include English code-switching (common in urban Pakistan).",
    },
    "ml": {
        "name": "Malayalam", "script": "malayalam",
        "chars": "Malayalam script. Combining vowel signs, chillu letters. Use text.find().",
        "cultural": "sadya, Onam, Vishu, boat race, toddy, chai",
        "places": "കൊച്ചി, തിരുവനന്തപുരം, കോഴിക്കോട്, മൂന്നാർ, ആലപ്പുഴ",
        "names": "രാജേഷ്, അനിൽ, സുരേഷ്, വിനോദ്, ലക്ഷ്മി, പ്രിയ, ദീപ, അജിത്, വിജയ്, മനോജ്",
        "family": "എന്റെ അമ്മ, എന്റെ അച്ഛൻ, എന്റെ ചേട്ടൻ, എന്റെ ചേച്ചി, അമ്മൂമ്മ, അപ്പൂപ്പൻ",
        "notes": "Include Manglish code-switching.",
    },
    "pa": {
        "name": "Punjabi (Gurmukhi)", "script": "gurmukhi",
        "chars": "Gurmukhi script. Tippi ੰ, bindi ਂ are combining. Use text.find().",
        "cultural": "ਲੱਸੀ, ਮੱਕੀ ਦੀ ਰੋਟੀ, ਸਰ੍ਹੋਂ ਦਾ ਸਾਗ, ਭੰਗੜਾ, ਗਿੱਧਾ, ਲੰਗਰ, ਗੁਰਦੁਆਰਾ",
        "places": "ਅੰਮ੍ਰਿਤਸਰ, ਲੁਧਿਆਣਾ, ਚੰਡੀਗੜ੍ਹ, ਜਲੰਧਰ, ਪਟਿਆਲਾ",
        "names": "ਗੁਰਪ੍ਰੀਤ, ਹਰਪ੍ਰੀਤ, ਅਮਰਜੀਤ, ਸਿਮਰਨ, ਜਸਪ੍ਰੀਤ, ਮਨਦੀਪ, ਰਾਜਵੀਰ, ਨਵਜੋਤ",
        "family": "ਮੇਰੀ ਮਾਂ, ਮੇਰੇ ਪਾਪਾ, ਮੇਰਾ ਵੀਰ, ਮੇਰੀ ਭੈਣ, ਦਾਦੀ, ਦਾਦਾ, ਸੱਸ",
        "notes": "ALWAYS Gurmukhi script, NEVER romanized. Include English code-switching.",
    },
    "es": {
        "name": "Spanish", "script": "latin",
        "chars": "á, é, í, ó, ú, ñ, ü, ¿, ¡ (NEVER substitute)",
        "cultural": "tapas, siesta, feria, botellón, nochevieja, quinceañera, sobremesa",
        "places": "Madrid, Barcelona, Sevilla, Valencia, Bilbao, Granada, Málaga",
        "names": "Carlos, Javier, Pablo, Alejandro, María, Carmen, Lucía, Sofía, Diego, Adrián",
        "family": "mi madre, mi padre, mi hermano, mi hermana, mi abuela, mi abuelo, mi suegra, mi cuñado",
    },
    "fr": {
        "name": "French", "script": "latin",
        "chars": "à, â, é, è, ê, ë, î, ï, ô, ù, û, ü, ÿ, ç, œ, æ (NEVER substitute)",
        "cultural": "apéro, boulangerie, grève, bac, CAF, mutuelle, galette des rois, fête des voisins",
        "places": "Paris, Lyon, Marseille, Toulouse, Bordeaux, Nantes, Strasbourg, Nice",
        "names": "Thomas, Nicolas, Julien, Antoine, Marie, Camille, Léa, Chloé, Maxime, Lucas",
        "family": "ma mère, mon père, mon frère, ma sœur, ma grand-mère, mon grand-père, ma belle-mère",
    },
    "it": {
        "name": "Italian", "script": "latin",
        "chars": "à, è, é, ì, ò, ù (NEVER substitute)",
        "cultural": "aperitivo, pranzo della domenica, ferragosto, nonna, bar, passeggiata, sagra",
        "places": "Roma, Milano, Napoli, Firenze, Torino, Bologna, Venezia, Palermo",
        "names": "Marco, Luca, Alessandro, Matteo, Giulia, Francesca, Chiara, Sara, Lorenzo, Andrea",
        "family": "mia madre, mio padre, mio fratello, mia sorella, mia nonna, mio nonno, mia suocera",
    },
    "nl": {
        "name": "Dutch", "script": "latin",
        "chars": "é, ë, ï, ö, ü (NEVER substitute, keep ij as two letters)",
        "cultural": "gezellig, borrel, Koningsdag, Sinterklaas, kroket, stamppot, fietspad",
        "places": "Amsterdam, Rotterdam, Utrecht, Den Haag, Eindhoven, Groningen, Maastricht",
        "names": "Jan, Pieter, Daan, Bram, Sophie, Fleur, Lotte, Sanne, Joost, Thijs",
        "family": "mijn moeder, mijn vader, mijn broer, mijn zus, oma, opa, schoonmoeder",
    },
    "pt": {
        "name": "Portuguese", "script": "latin",
        "chars": "á, â, ã, à, é, ê, í, ó, ô, õ, ú, ç (NEVER substitute). Mix BR and PT-PT.",
        "cultural": "churrasco, saudade, padaria, novela, carnaval, feijoada, pastel de nata, festa junina",
        "places": "São Paulo, Rio de Janeiro, Lisboa, Porto, Brasília, Salvador, Curitiba",
        "names": "João, Pedro, Rafael, Lucas, Maria, Ana, Beatriz, Juliana, Gustavo, Matheus",
        "family": "minha mãe, meu pai, meu irmão, minha irmã, minha avó, meu avô, minha sogra",
    },
    "pl": {
        "name": "Polish", "script": "latin",
        "chars": "ą, ć, ę, ł, ń, ó, ś, ź, ż (NEVER substitute)",
        "cultural": "wigilia, imieniny, działka, zapiekanka, żurek, oscypek, majówka",
        "places": "Warszawa, Kraków, Gdańsk, Wrocław, Poznań, Łódź, Zakopane",
        "names": "Tomasz, Krzysztof, Piotr, Jakub, Anna, Katarzyna, Agnieszka, Magda, Mateusz, Kacper",
        "family": "moja mama, mój tata, mój brat, moja siostra, babcia, dziadek, teściowa",
    },
    "uk": {
        "name": "Ukrainian", "script": "cyrillic",
        "chars": "Cyrillic: а-я + і, ї, є, ґ. NEVER Russian и instead of і. Use text.find().",
        "cultural": "борщ, вареники, вишиванка, Різдво, Великдень, козак, гривня",
        "places": "Київ, Львів, Одеса, Харків, Дніпро, Запоріжжя, Івано-Франківськ",
        "names": "Олександр, Андрій, Дмитро, Тарас, Оксана, Наталія, Ірина, Олена, Богдан, Ярослав",
        "family": "моя мама, мій тато, мій брат, моя сестра, бабуся, дідусь, свекруха",
        "notes": "ALWAYS Ukrainian Cyrillic, NEVER Russian. Include surzhyk code-switching where natural.",
    },
    "ru": {
        "name": "Russian", "script": "cyrillic",
        "chars": "Cyrillic: а-я + ё (NEVER substitute е for ё). Use text.find().",
        "cultural": "дача, баня, шашлык, Новый год, Масленица, пельмени, электричка",
        "places": "Москва, Санкт-Петербург, Казань, Новосибирск, Екатеринбург, Сочи",
        "names": "Алексей, Дмитрий, Сергей, Андрей, Мария, Екатерина, Анна, Ольга, Максим, Иван",
        "family": "мама, папа, брат, сестра, бабушка, дедушка, свекровь, тёща",
    },
    "hi": {
        "name": "Hindi", "script": "devanagari",
        "chars": "Devanagari: अ-ह + matras. Include Hinglish code-switching. Use text.find().",
        "cultural": "दिवाली, होली, शादी, चाय, क्रिकेट, बॉलीवुड, मंदिर, बाज़ार",
        "places": "दिल्ली, मुंबई, बेंगलुरु, जयपुर, वाराणसी, कोलकाता, चेन्नई",
        "names": "राहुल, अमित, विकास, प्रिया, नेहा, अंजलि, रोहन, आदित्य, पूजा, दीपिका",
        "family": "मेरी माँ, मेरे पापा, मेरा भाई, मेरी बहन, दादी, नानी, सास",
        "notes": "Mix Devanagari + English naturally (Hinglish). Romanized Hindi also OK in casual speech.",
    },
    "ja": {
        "name": "Japanese", "script": "cjk",
        "chars": "Kanji + hiragana + katakana. Use text.find() for offsets.",
        "cultural": "花見, 忘年会, お盆, 新年会, 居酒屋, コンビニ, 部活, バイト",
        "places": "東京, 大阪, 京都, 横浜, 名古屋, 札幌, 福岡, 沖縄",
        "names": "太郎, 花子, 健太, 美咲, 大輔, さくら, 翔太, 結衣, 拓也, 陽菜",
        "family": "お母さん, お父さん, 兄, 姉, おばあちゃん, おじいちゃん, 義母",
        "notes": "Use casual Japanese (タメ口), not keigo. Include honorific suffixes (さん, ちゃん, くん).",
    },
    "ko": {
        "name": "Korean", "script": "cjk",
        "chars": "Hangul. Use text.find() for offsets.",
        "cultural": "치맥, 노래방, PC방, 수능, 설날, 추석, 삼겹살, 소맥",
        "places": "서울, 부산, 제주도, 인천, 대구, 광주, 경주, 강릉",
        "names": "민수, 지훈, 현우, 서연, 수진, 지은, 태현, 유진, 준호, 하은",
        "family": "엄마, 아빠, 형/오빠, 누나/언니, 할머니, 할아버지, 시어머니",
        "notes": "Use 반말 (casual) register. Include Korean honorific levels where natural.",
    },
    "zh": {
        "name": "Chinese (Simplified)", "script": "cjk",
        "chars": "Simplified Chinese characters. Use text.find() for offsets.",
        "cultural": "春节, 中秋, 火锅, 奶茶, 高考, 红包, 广场舞, 双十一",
        "places": "北京, 上海, 广州, 深圳, 成都, 杭州, 西安, 南京",
        "names": "小明, 小红, 张伟, 王芳, 李强, 刘洋, 陈静, 赵磊, 周婷, 吴鹏",
        "family": "我妈, 我爸, 哥哥, 姐姐, 奶奶, 外婆, 爷爷, 外公, 婆婆",
        "notes": "Use casual spoken Mandarin, not literary Chinese. Include internet slang where natural.",
    },
}


def build_prompt(lang: str, count: int = 10, mode: str = "tuning", batch_num: int = 1, provider: str = "claude") -> str:
    cfg = LANG_CONFIG[lang]
    name = cfg["name"]

    if mode == "tuning":
        id_prefix = f"{lang}_{provider}_t"
        id_range = f"{id_prefix}_001 through {id_prefix}_{count:03d}"
        out_file = f"/home/ubuntu/projects/oneiron-ner/data/raw/silver_synthetic/{lang}_{provider}_tuning.jsonl"
    else:
        id_prefix = f"{lang}_{provider}_b{batch_num}"
        id_range = f"{id_prefix}_001 through {id_prefix}_{count:03d}"
        out_file = f"/home/ubuntu/projects/oneiron-ner/data/raw/silver_synthetic/{lang}_{provider}_batch{batch_num}.jsonl"

    script_notes = ""
    if cfg["script"] in ("bengali", "tamil", "telugu", "malayalam", "gurmukhi"):
        script_notes = f"\n\nCRITICAL: {name} has combining characters. Visual chars ≠ codepoints. Use text.find(surface) and len(surface) for offsets."
    elif cfg["script"] in ("hebrew", "urdu", "persian"):
        script_notes = f"\n\nCRITICAL: {name} is RTL. Python offsets are LTR in memory — use text.find(surface) normally."
    elif cfg["script"] == "cyrillic":
        script_notes = "\n\nCRITICAL: NEVER mix Cyrillic with Latin lookalike characters."
    elif cfg["script"] == "cyrillic+latin":
        split = count // 2
        script_notes = f"\n\nCRITICAL: Serbian is digraphic. Generate {split} Cyrillic + {count - split} Latin convos. NEVER mix scripts in one conversation."

    extra = f"\n{cfg['notes']}" if cfg.get("notes") else ""

    provider_note = ""
    if provider == "gpt54":
        provider_note = "\n\nIMPORTANT: Use PROPER Unicode characters for this language. Do NOT use ASCII substitutes for accented/special characters. Write the output file using the Write tool, then verify with Bash."

    return f"""Generate exactly {count} {name} ({lang}) NER-annotated conversations and save to `{out_file}`.

## JSONL Format (one JSON per line, {count} lines total)
```json
{{"source": "synthetic_{lang}_{provider}", "source_id": "{id_prefix}_001", "language": "{lang}", "format": "conversation", "turns": [{{"speaker": "A", "text": "..."}}, {{"speaker": "B", "text": "..."}}], "entities": [{{"surface": "...", "type": "...", "start": 0, "end": 5, "turn_index": 0}}]}}
```
source_id: {id_range}.

## Entity Types
{ENTITY_TYPES}
{script_notes}

## {name} Guidelines
- Casual spoken {name}, informal register. Proper characters: {cfg['chars']}.
- Cultural: {cfg['cultural']}
- Places: {cfg['places']}
- Names: {cfg['names']}
- Family terms: {cfg['family']}
- 3-6 turns, 5-10 entities per conversation. DIVERSE topics. All 9 base types across all conversations.{extra}

## Computing Offsets
For EVERY entity: `start = text.find(surface)`, `end = start + len(surface)`, verify `text[start:end] == surface`.

## Verification (MUST RUN)
```python
import json
path = "{out_file}"
bad = 0; total = 0
with open(path) as f:
    for line in f:
        rec = json.loads(line)
        for ent in rec["entities"]:
            total += 1
            text = rec["turns"][ent["turn_index"]]["text"]
            if text[ent["start"]:ent["end"]] != ent["surface"]:
                print(f"BAD: '{{ent['surface']}}' != '{{text[ent['start']:ent['end']]}}' in {{rec['source_id']}}")
                bad += 1
print(f"\\n{{total}} entities, {{bad}} bad offsets, {{sum(1 for _ in open(path))}} records")
```
Fix and re-verify until 0 bad offsets. Do NOT proceed until clean.{provider_note}"""


ALL_NEW_LANGS = list(LANG_CONFIG.keys())
WAVE1_LATIN = ["da", "no", "fi", "hr", "hu", "sk", "et", "lt", "lv", "ca", "af", "tl", "ms", "sw"]
WAVE2_COMPLEX = ["el", "he", "fa", "bg", "sr", "bn", "ta", "te", "ur", "ml", "pa"]


if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else "da"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    mode = sys.argv[3] if len(sys.argv) > 3 else "tuning"
    provider = sys.argv[4] if len(sys.argv) > 4 else "claude"
    print(build_prompt(lang, count, mode, provider=provider))
