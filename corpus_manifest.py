"""
The Field Guide — RAG corpus ingestion manifest
===========================================================

Scope: UK-region naturalist assistant (plants, fungi, insects, birds),
matching the project brief: ~280 species accounts + regional status lists
(invasive / protected / toxic) for UK.

IMPORTANT — verify before trusting for the safety-critical parts of this app:
- Species names and general biology below are common knowledge, low
  hallucination risk.
- Exact legal status (which Schedule, which Act, current invasive-list
  membership) must be pulled live from the government sources listed in
  REGIONAL_STATUS_SOURCES at ingestion time -- do not hardcode "this species
  is protected/invasive" anywhere in your pipeline from this list alone.
- Identifier fields below (isbn, doi, arxiv_id, wikipedia_title, url) are
  left as None wherever unverified -- a wrong ID is worse than no ID, since
  the extractor's fallback logic already resolves these dynamically at
  runtime.
- CAUTION on the PAPERS doi fields below: these were supplied as "real
  identifiers where known", but two were spot-checked against live search
  results and neither matched the claimed paper (wrong author/year/journal
  for the knotweed DOI; no matching paper found at all for the amatoxin
  DOI). Treat every doi value in PAPERS as UNVERIFIED until confirmed
  against Crossref/the publisher directly -- do not rely on them for the
  toxicology-related entries in particular, given the safety stakes.

NOTE on Wikispecies (named in the project brief alongside Wikipedia/GBIF):
Verified via search this session -- it is real, with its own MediaWiki API
at https://species.wikimedia.org/w/api.php (separate from en.wikipedia.org).
No entries have been added to SPECIES_WIKI for it because the existing
02_BooksArticles-Extractor notebook's Wikipedia cell only calls
en.wikipedia.org's API -- it has no Wikispecies-specific extraction logic
yet. Add a "wikispecies" source_type group (or extend SPECIES_WIKI with a
source_type override) once that extractor path exists, rather than adding
placeholder entries an extractor can't yet resolve.

NOTE on GBIF as a knowledge-corpus source (vs. distribution data): GBIF's
main /v1/species/{taxonKey} endpoint (verified via search) returns
taxonomic/name-matching data (rank, classification, synonyms), not
Wikipedia-style narrative species accounts. GBIF does have a separate
/v1/species/{taxonKey}/descriptions endpoint that can carry prose, but
coverage is sparse and inconsistent per species -- don't treat GBIF as a
narrative-account source equivalent to SPECIES_WIKI. Its existing use in
REGIONAL_STATUS_SOURCES (distribution_data category, confirming a species
is realistically present in the user's region) remains the accurate role
for it here.

Schema per group (all groups share title/category/source_type/notes; each
adds one or more fields matching its extractor's fallback logic):
- SPECIES_WIKI:            {..., "wikipedia_title": str | None}
- REGIONAL_STATUS_SOURCES:  {..., "url": str | None}
- BOOKS:                    {..., "isbn": str | None}
- PAPERS:                   {..., "doi": str | None, "arxiv_id": str | None}
- VIDEOS:                   {..., "url": str | None, "video_id": str | None}
  -> url/video_id are ALL None as of this manifest version -- no web search
     was available when this group was created, so no actual video was
     looked up or verified. Do not treat any entry here as a real,
     checked video until url/video_id are filled in.
"""

# ---------------------------------------------------------------------------
# 1. SPECIES WIKIPEDIA ACCOUNTS
#    wikipedia_title fields have been set to the common name (without the
#    parenthetical binomial) to increase Wikipedia API match rate.
# ---------------------------------------------------------------------------

SPECIES_WIKI = [
    # --- Invasive plants (commonly flagged in UK) ---
    {"title": "Japanese knotweed (Reynoutria japonica)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Flagship UK invasive; legal duty not to spread.", "wikipedia_title": "Japanese knotweed"},
    {"title": "Himalayan balsam (Impatiens glandulifera)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Riverbank invasive.", "wikipedia_title": "Himalayan balsam"},
    {"title": "Giant hogweed (Heracleum mantegazzianum)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Also a phytotoxic/skin-burn hazard — dual invasive+toxic entry.", "wikipedia_title": "Giant hogweed"},
    {"title": "Rhododendron ponticum", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Invasive in woodland/heath; also toxic to livestock.", "wikipedia_title": "Rhododendron ponticum"},
    {"title": "New Zealand pigmyweed (Crassula helmsii)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Aquatic invasive.", "wikipedia_title": "New Zealand pigmyweed"},
    {"title": "Floating pennywort (Hydrocotyle ranunculoides)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Aquatic invasive.", "wikipedia_title": "Floating pennywort"},
    {"title": "Parrot's feather (Myriophyllum aquaticum)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Aquatic invasive.", "wikipedia_title": "Parrot's feather"},
    {"title": "Water fern / fairy fern (Azolla filiculoides)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Aquatic invasive.", "wikipedia_title": "Azolla filiculoides"},
    {"title": "American skunk cabbage (Lysichiton americanus)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Wetland invasive. wikipedia_title corrected to the binomial -- the common-name title returned an empty extract (VERIFIED via search this session: en.wikipedia.org/wiki/Lysichiton_americanus is the real article; 'American skunk cabbage' is a common-name redirect that resolved to no readable text).", "wikipedia_title": "Lysichiton americanus"},
    {"title": "Cherry laurel (Prunus laurocerasus)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Woodland invasive; foliage toxic (cyanogenic).", "wikipedia_title": "Cherry laurel"},
    {"title": "Cotoneaster (various spp.)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Several Cotoneaster species listed invasive in UK.", "wikipedia_title": "Cotoneaster"},
    {"title": "Spanish bluebell (Hyacinthoides hispanica)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Hybridizes with/displaces native bluebell — good 'lookalike' teaching case.", "wikipedia_title": "Spanish bluebell"},
    {"title": "Water primrose (Ludwigia grandiflora)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Aquatic invasive, rapidly colonises still water.", "wikipedia_title": "Water primrose"},
    {"title": "Fanwort (Cabomba caroliniana)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Submerged aquatic, invasive in ponds and lakes.", "wikipedia_title": "Fanwort"},
    {"title": "Curly waterweed (Lagarosiphon major)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Oxygen weed, invasive in UK waterways.", "wikipedia_title": "Curly waterweed"},
    {"title": "Nuttall's waterweed (Elodea nuttallii)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Common invasive aquatic plant.", "wikipedia_title": "Nuttall's waterweed"},
    {"title": "Himalayan honeysuckle (Leycesteria formosa)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Garden escapee, invasive in scrub and woodland edges.", "wikipedia_title": "Himalayan honeysuckle"},
    {"title": "Wall cotoneaster (Cotoneaster horizontalis)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Invasive on cliffs and walls, bird-dispersed.", "wikipedia_title": "Wall cotoneaster"},

    # --- Native bluebell + common lookalike pairs (good for stretch goal) ---
    {"title": "Common bluebell (Hyacinthoides non-scripta)", "category": "native_protected_plant", "source_type": "wikipedia", "notes": "Protected against uprooting/sale under Wildlife and Countryside Act.", "wikipedia_title": "Common bluebell"},

    # --- Protected UK plants ---
    {"title": "Lady's slipper orchid (Cypripedium calceolus)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Extremely rare, heavily protected.", "wikipedia_title": "Lady's slipper orchid"},
    {"title": "Fen orchid (Liparis loeselii)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Schedule-protected.", "wikipedia_title": "Fen orchid"},
    {"title": "Snowdon lily (Gagea serotina)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Endemic, protected.", "wikipedia_title": "Snowdon lily"},
    {"title": "Early spider orchid (Ophrys sphegodes)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Protected orchid.", "wikipedia_title": "Early spider orchid"},
    {"title": "Military orchid (Orchis militaris)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Rare, protected.", "wikipedia_title": "Military orchid"},
    {"title": "Ghost orchid (Epipogium aphyllum)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Extremely rare/possibly extinct in UK; protected.", "wikipedia_title": "Ghost orchid"},
    {"title": "Deptford pink (Dianthus armeria)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Rare, Schedule 8; verify current status.", "wikipedia_title": "Deptford pink"},
    {"title": "Red helleborine (Cephalanthera rubra)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Very rare, protected.", "wikipedia_title": "Red helleborine"},
    {"title": "Tower mustard (Arabis glabra)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Nationally scarce, Schedule 8.", "wikipedia_title": "Tower mustard"},
    {"title": "Spotted cat's-ear (Hypochaeris maculata)", "category": "protected_plant", "source_type": "wikipedia", "notes": "Rare, protected. wikipedia_title corrected to the binomial -- the common-name title returned an empty extract (VERIFIED via search this session: en.wikipedia.org/wiki/Hypochaeris_maculata is the real article).", "wikipedia_title": "Hypochaeris maculata"},
    {"title": "Purple saxifrage (Saxifraga oppositifolia)", "category": "notable_plant", "source_type": "wikipedia", "notes": "Arctic-alpine and locally rare; do not state statutory protection without a region-specific legal check.", "wikipedia_title": "Purple saxifrage"},

    # --- Toxic / poisonous plants (RHS list territory) ---
    {"title": "Foxglove (Digitalis purpurea)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Cardiac glycoside toxicity.", "wikipedia_title": "Foxglove"},
    {"title": "Deadly nightshade (Atropa belladonna)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Highly toxic, tropane alkaloids.", "wikipedia_title": "Deadly nightshade"},
    {"title": "Hemlock (Conium maculatum)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Fatal poisoning risk; classic hemlock/cow parsley confusion case.", "wikipedia_title": "Hemlock"},
    {"title": "Yew (Taxus baccata)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Foliage/seeds toxic to humans and livestock; berries deceptively edible-looking.", "wikipedia_title": "Yew"},
    {"title": "Laburnum (Laburnum anagyroides)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Seeds highly toxic, resemble pea pods — common garden hazard.", "wikipedia_title": "Laburnum"},
    {"title": "Monkshood / wolfsbane (Aconitum napellus)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "One of the most toxic garden plants.", "wikipedia_title": "Monkshood"},
    {"title": "Oleander (Nerium oleander)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Cardiac glycosides; common ornamental.", "wikipedia_title": "Oleander"},
    {"title": "Lily of the valley (Convallaria majalis)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Cardiac glycosides; pretty/deceptive.", "wikipedia_title": "Lily of the valley"},
    {"title": "Ivy (Hedera helix)", "category": "mild_toxicity_plant", "source_type": "wikipedia", "notes": "Skin irritant / mild GI toxicity, ubiquitous.", "wikipedia_title": "Ivy"},
    {"title": "Daffodil (Narcissus spp.)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Bulbs toxic, mistaken for onions historically.", "wikipedia_title": "Daffodil"},
    {"title": "Autumn crocus (Colchicum autumnale)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Colchicine poisoning; leaves resemble wild garlic — classic fatal confusion case.", "wikipedia_title": "Autumn crocus"},
    {"title": "Wild garlic / ramsons (Allium ursinum)", "category": "edible_plant_lookalike", "source_type": "wikipedia", "notes": "Non-toxic but frequently confused with lily-of-the-valley and autumn crocus — key lookalike-warning entry.", "wikipedia_title": "Wild garlic"},
    {"title": "Cow parsley (Anthriscus sylvestris)", "category": "edible_plant_lookalike", "source_type": "wikipedia", "notes": "Non-toxic but visually similar to hemlock/giant hogweed.", "wikipedia_title": "Cow parsley"},
    {"title": "Bittersweet (Solanum dulcamara)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Toxic, though less potent; red berries are attractive to children.", "wikipedia_title": "Bittersweet"},
    {"title": "Thorn apple (Datura stramonium)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Highly toxic, tropane alkaloids; occasionally found as a weed.", "wikipedia_title": "Thorn apple"},
    {"title": "Castor bean (Ricinus communis)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Extremely toxic; ornamental, but rare in UK gardens.", "wikipedia_title": "Castor bean"},
    {"title": "Buttercup (Ranunculus spp.)", "category": "mild_toxicity_plant", "source_type": "wikipedia", "notes": "Skin irritant and GI upset if eaten; common meadow plant.", "wikipedia_title": "Buttercup"},
    {"title": "Spurge (Euphorbia spp.)", "category": "mild_toxicity_plant", "source_type": "wikipedia", "notes": "Milky sap is skin irritant and toxic if ingested.", "wikipedia_title": "Spurge"},
    {"title": "Common ragwort (Jacobaea vulgaris)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "Toxic to livestock; management is subject to ragwort-control guidance in England and Wales. Do not describe it as generally 'notifiable'.", "wikipedia_title": "Common ragwort"},

    # --- Toxic / dangerous fungi ---
    {"title": "Death cap (Amanita phalloides)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Responsible for most fatal mushroom poisonings worldwide.", "wikipedia_title": "Death cap"},
    {"title": "Destroying angel (Amanita virosa)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Lethal amatoxins; resembles edible white mushrooms.", "wikipedia_title": "Destroying angel"},
    {"title": "Fly agaric (Amanita muscaria)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Psychoactive/toxic; iconic red-cap species, high misID risk.", "wikipedia_title": "Fly agaric"},
    {"title": "Panther cap (Amanita pantherina)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Toxic, resembles edible Amanita rubescens.", "wikipedia_title": "Panther cap"},
    {"title": "Deadly webcap (Cortinarius rubellus)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Delayed fatal kidney toxicity; easily confused with edible chanterelles by novices.", "wikipedia_title": "Deadly webcap"},
    {"title": "Yellow stainer (Agaricus xanthodermus)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "GI toxin; common lookalike for field/horse mushrooms.", "wikipedia_title": "Yellow stainer"},
    {"title": "False morel (Gyromitra esculenta)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Toxic despite the species name; resembles true morels.", "wikipedia_title": "False morel"},
    {"title": "Sulphur tuft (Hypholoma fasciculare)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Bitter/toxic; common on stumps, confused with edible Armillaria.", "wikipedia_title": "Sulphur tuft"},
    {"title": "Fool's mushroom (Amanita verna)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Similar to destroying angel; contains amatoxins.", "wikipedia_title": "Fool's mushroom"},
    {"title": "Deadly dapperling (Lepiota cristata)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Small brown mushrooms, toxic; common in grasslands.", "wikipedia_title": "Deadly dapperling"},
    {"title": "Brown roll-rim (Paxillus involutus)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Toxic when raw; can cause haemolytic reactions.", "wikipedia_title": "Brown roll-rim"},
    {"title": "Cortinarius orellanus (fool's webcap)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Contains orellanine; delayed kidney failure.", "wikipedia_title": "Cortinarius orellanus"},
    {"title": "Inocybe (many species)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Many Inocybe contain muscarine; toxic.", "wikipedia_title": "Inocybe"},

    # --- Common edible/harmless fungi for contrast (lookalike pairs) ---
    {"title": "Field mushroom (Agaricus campestris)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Confused with yellow stainer and young death cap.", "wikipedia_title": "Field mushroom"},
    {"title": "Chanterelle (Cantharellus cibarius)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Confused with false chanterelle and deadly webcap.", "wikipedia_title": "Chanterelle"},
    {"title": "Common morel (Morchella esculenta)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Confused with toxic false morel.", "wikipedia_title": "Common morel"},
    {"title": "Honey fungus (Armillaria mellea)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Edible when cooked, confused with sulphur tuft; also a tree pathogen.", "wikipedia_title": "Honey fungus"},
    {"title": "Horse mushroom (Agaricus arvensis)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Edible; confused with yellow stainer.", "wikipedia_title": "Horse mushroom"},
    {"title": "Wood blewit (Lepista nuda)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Do not treat edibility as a safe identification claim; can be confused with purple Cortinarius species. Current widely used name is Lepista nuda.", "wikipedia_title": "Wood blewit"},
    {"title": "Penny bun / porcini (Boletus edulis)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Highly prized edible; may be confused with bitter boletes.", "wikipedia_title": "Penny bun"},
    {"title": "Hedgehog mushroom (Hydnum repandum)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Edible, distinctive toothy underside; no toxic lookalikes in UK.", "wikipedia_title": "Hedgehog mushroom"},

    # --- Invasive / notable invertebrates ---
    {"title": "Asian hornet (Vespa velutina)", "category": "invasive_insect", "source_type": "wikipedia", "notes": "Reportable UK invasive; threat to honeybees.", "wikipedia_title": "Asian hornet"},
    {"title": "Harlequin ladybird (Harmonia axyridis)", "category": "invasive_insect", "source_type": "wikipedia", "notes": "Displaces native ladybirds; common misID target.", "wikipedia_title": "Harlequin ladybird"},
    {"title": "Oak processionary moth (Thaumetopoea processionea)", "category": "invasive_insect", "source_type": "wikipedia", "notes": "Caterpillar hairs cause skin/respiratory irritation — safety-relevant.", "wikipedia_title": "Oak processionary moth"},
    {"title": "Box tree moth (Cydalima perspectalis)", "category": "invasive_insect", "source_type": "wikipedia", "notes": "Garden pest, rapidly spreading in UK.", "wikipedia_title": "Box tree moth"},
    {"title": "Signal crayfish (Pacifastacus leniusculus)", "category": "invasive_invertebrate", "source_type": "wikipedia", "notes": "Aquatic invasive, displaces native white-clawed crayfish.", "wikipedia_title": "Signal crayfish"},
    {"title": "Zebra mussel (Dreissena polymorpha)", "category": "invasive_invertebrate", "source_type": "wikipedia", "notes": "Freshwater invasive.", "wikipedia_title": "Zebra mussel"},
    {"title": "Chinese mitten crab (Eriocheir sinensis)", "category": "invasive_invertebrate", "source_type": "wikipedia", "notes": "Invasive in UK rivers; burrows damage banks.", "wikipedia_title": "Chinese mitten crab"},
    {"title": "American mink (Neogale vison)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Non-native predator established in Britain; current genus is Neogale.", "wikipedia_title": "American mink"},

    # --- Dangerous/venomous UK species (safety relevance) ---
    {"title": "European adder (Vipera berus)", "category": "venomous_animal", "source_type": "wikipedia", "notes": "Only venomous snake native to UK — dog-safety relevance mentioned in brief.", "wikipedia_title": "European adder"},
    {"title": "False widow spider (Steatoda nobilis)", "category": "venomous_animal", "source_type": "wikipedia", "notes": "UK's most medically significant spider; frequently misidentified as dangerous when harmless spiders are found.", "wikipedia_title": "False widow spider"},
    {"title": "European hornet (Vespa crabro)", "category": "native_animal_lookalike", "source_type": "wikipedia", "notes": "Native UK hornet; can sting and should not be described categorically as non-aggressive. Important comparison species for distinguishing from yellow-legged/Asian hornet.", "wikipedia_title": "European hornet"},

    # --- Protected UK animals ---
    {"title": "Great crested newt (Triturus cristatus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Heavily protected; development-law relevance.", "wikipedia_title": "Great crested newt"},
    {"title": "Natterjack toad (Epidalea calamita)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Rare, protected.", "wikipedia_title": "Natterjack toad"},
    {"title": "Sand lizard (Lacerta agilis)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected reptile.", "wikipedia_title": "Sand lizard"},
    {"title": "Smooth snake (Coronella austriaca)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Rarest UK snake, protected.", "wikipedia_title": "Smooth snake"},
    {"title": "Water vole (Arvicola amphibius)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected, declining rapidly.", "wikipedia_title": "Water vole"},
    {"title": "Hazel dormouse (Muscardinus avellanarius)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected, indicator species.", "wikipedia_title": "Hazel dormouse"},
    {"title": "Pine marten (Martes martes)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected mustelid.", "wikipedia_title": "Pine marten"},
    {"title": "European otter (Lutra lutra)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected.", "wikipedia_title": "European otter"},
    {"title": "All UK bat species (Chiroptera)", "category": "protected_animal", "source_type": "wikipedia", "notes": "All UK bats are protected — worth a dedicated ingestion note, not just one species page.", "wikipedia_title": "Bat"},
    {"title": "Red squirrel (Sciurus vulgaris)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected, threatened by invasive grey squirrel.", "wikipedia_title": "Red squirrel"},
    {"title": "Grey squirrel (Sciurus carolinensis)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Invasive, displaces red squirrel — key contrast pair.", "wikipedia_title": "Grey squirrel"},
    {"title": "Badger (Meles meles)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected under the Protection of Badgers Act 1992, not Schedule 5 but still legally protected. wikipedia_title corrected -- plain 'Badger' resolves to the broader family-level article (all badger species worldwide, including American and honey badgers), not the specific UK/European species (VERIFIED via search this session: en.wikipedia.org/wiki/European_badger is Meles meles specifically).", "wikipedia_title": "European badger"},
    {"title": "Grass snake (Natrix helvetica)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected from killing/injury; common in wetlands.", "wikipedia_title": "Grass snake"},
    {"title": "Common lizard (Zootoca vivipara)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected from killing/injury; widespread.", "wikipedia_title": "Common lizard"},
    
    # --- Common UK garden/wild birds (for general ID accuracy, non-toxic/legal-status baseline) ---
    {"title": "European robin (Erithacus rubecula)", "category": "common_bird", "source_type": "wikipedia", "notes": "Baseline common species for accuracy testing.", "wikipedia_title": "European robin"},
    {"title": "Blackbird (Turdus merula)", "category": "common_bird", "source_type": "wikipedia", "notes": "Baseline common species.", "wikipedia_title": "Blackbird"},
    {"title": "Blue tit (Cyanistes caeruleus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Baseline common species.", "wikipedia_title": "Blue tit"},
    {"title": "House sparrow (Passer domesticus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Baseline common species, declining.", "wikipedia_title": "House sparrow"},
    {"title": "Wood pigeon (Columba palumbus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Baseline common species.", "wikipedia_title": "Wood pigeon"},
    {"title": "Eurasian magpie (Pica pica)", "category": "common_bird", "source_type": "wikipedia", "notes": "Baseline common species.", "wikipedia_title": "Eurasian magpie"},
    {"title": "Barn owl (Tyto alba)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected (special penalties for disturbance).", "wikipedia_title": "Barn owl"},
    {"title": "Kingfisher (Alcedo atthis)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected. wikipedia_title corrected -- plain 'Kingfisher' resolves to the family-level article (Alcedinidae, dozens of species worldwide), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Common_kingfisher is Alcedo atthis specifically).", "wikipedia_title": "Common kingfisher"},
    {"title": "Peregrine falcon (Falco peregrinus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected.", "wikipedia_title": "Peregrine falcon"},
    {"title": "Chaffinch (Fringilla coelebs)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common woodland and garden bird. wikipedia_title corrected -- plain 'Chaffinch' is a genuine Wikipedia DISAMBIGUATION page (7 chaffinch species worldwide), which has no plain-text extract by design (VERIFIED via search this session: en.wikipedia.org/wiki/Eurasian_chaffinch is Fringilla coelebs specifically).", "wikipedia_title": "Eurasian chaffinch"},
    {"title": "Great tit (Parus major)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common garden bird.", "wikipedia_title": "Great tit"},
    {"title": "Starling (Sturnus vulgaris)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common, but declining; notable murmurations. wikipedia_title corrected -- plain 'Starling' resolves to the family-level article (Sturnidae, 128 species worldwide), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Common_starling is Sturnus vulgaris specifically).", "wikipedia_title": "Common starling"},
    {"title": "Song thrush (Turdus philomelos)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common, but declining; lovely song.", "wikipedia_title": "Song thrush"},
    {"title": "Carrion crow (Corvus corone)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common corvid.", "wikipedia_title": "Carrion crow"},
    {"title": "Jackdaw (Corvus monedula)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common corvid. wikipedia_title corrected -- plain 'Jackdaw' resolves to the genus-level article (Coloeus, 2 species), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Western_jackdaw is Coloeus monedula specifically).", "wikipedia_title": "Western jackdaw"},
    {"title": "Rook (Corvus frugilegus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common corvid. wikipedia_title corrected -- plain 'Rook' is a genuine Wikipedia DISAMBIGUATION page (shared with chess pieces, aircraft, a place name), which has no plain-text extract by design (VERIFIED via search this session: en.wikipedia.org/wiki/Rook_(bird) is the real article, but MediaWiki's redirects=1 does not follow disambig pages to a specific target).", "wikipedia_title": "Rook (bird)"},
    {"title": "Common swift (Apus apus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Summer migrant, common.", "wikipedia_title": "Common swift"},
    {"title": "House martin (Delichon urbicum)", "category": "common_bird", "source_type": "wikipedia", "notes": "Summer migrant; current scientific name is Delichon urbicum.", "wikipedia_title": "House martin"},
    {"title": "Swallow (Hirundo rustica)", "category": "common_bird", "source_type": "wikipedia", "notes": "Summer migrant, common. wikipedia_title corrected -- plain 'Swallow' resolves to the family-level article (Hirundinidae, ~90 species worldwide), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Barn_swallow is Hirundo rustica specifically).", "wikipedia_title": "Barn swallow"},
    {"title": "Goldfinch (Carduelis carduelis)", "category": "common_bird", "source_type": "wikipedia", "notes": "Colourful garden bird. wikipedia_title corrected -- plain 'Goldfinch' is a genuine Wikipedia DISAMBIGUATION page (shared with American finches, a painting, a novel, and its film adaptation), which has no plain-text extract by design (VERIFIED via search this session: en.wikipedia.org/wiki/European_goldfinch is Carduelis carduelis specifically).", "wikipedia_title": "European goldfinch"},
    {"title": "Greenfinch (Chloris chloris)", "category": "common_bird", "source_type": "wikipedia", "notes": "Garden bird, declining due to disease. wikipedia_title corrected -- plain 'Greenfinch' resolves to the genus-level article (Chloris, 6 species), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/European_greenfinch is Chloris chloris specifically).", "wikipedia_title": "European greenfinch"},

    # --- Common garden insects/pollinators for baseline accuracy ---
    {"title": "Seven-spot ladybird (Coccinella septempunctata)", "category": "common_insect", "source_type": "wikipedia", "notes": "Native contrast to invasive harlequin.", "wikipedia_title": "Seven-spot ladybird"},
    {"title": "Red admiral (Vanessa atalanta)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common butterfly baseline.", "wikipedia_title": "Red admiral"},
    {"title": "Peacock butterfly (Aglais io)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common butterfly baseline.", "wikipedia_title": "Peacock butterfly"},
    {"title": "Buff-tailed bumblebee (Bombus terrestris)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common pollinator baseline.", "wikipedia_title": "Buff-tailed bumblebee"},
    {"title": "Common wasp (Vespula vulgaris)", "category": "common_insect", "source_type": "wikipedia", "notes": "Frequently confused with hornets — disambiguation value.", "wikipedia_title": "Common wasp"},
    {"title": "Emperor moth (Saturnia pavonia)", "category": "common_insect", "source_type": "wikipedia", "notes": "VERIFIED via search this session. Britain's only native Saturniidae; common on moorland/heathland, non-toxic, no protected/invasive status -- a good baseline non-flagged ID case. Matches the misidentification example already used in the app's own test transcript.", "wikipedia_title": "Saturnia pavonia"},
    {"title": "Small tortoiseshell (Aglais urticae)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common butterfly.", "wikipedia_title": "Small tortoiseshell"},
    {"title": "Meadow brown (Maniola jurtina)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common grassland butterfly.", "wikipedia_title": "Meadow brown"},
    {"title": "Common blue (Polyommatus icarus)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common blue butterfly.", "wikipedia_title": "Common blue"},
    {"title": "Honey bee (Apis mellifera)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common pollinator, not native but widely managed.", "wikipedia_title": "Honey bee"},
    {"title": "Red-tailed bumblebee (Bombus lapidarius)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common bumblebee.", "wikipedia_title": "Red-tailed bumblebee"},
    {"title": "Garden spider (Araneus diadematus)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common orb-weaver, harmless.", "wikipedia_title": "Garden spider"},
    {"title": "European earwig (Forficula auricularia)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common garden insect.", "wikipedia_title": "European earwig"},
    {"title": "Green shieldbug (Palomena prasina)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common shieldbug, changes colour in autumn.", "wikipedia_title": "Green shieldbug"},
    {"title": "Common field grasshopper (Chorthippus brunneus)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common grasshopper.", "wikipedia_title": "Common field grasshopper"},
    {"title": "Daddy longlegs / crane fly (Tipula spp.)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common, often mistaken for giant mosquitoes.", "wikipedia_title": "Crane fly"},

    # --- Common wildflowers (non-toxic, non-invasive) ---
    {"title": "Daisy (Bellis perennis)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Ubiquitous lawn flower.", "wikipedia_title": "Daisy"},
    {"title": "Dandelion (Taraxacum officinale)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Common weed; leaves edible.", "wikipedia_title": "Dandelion"},
    {"title": "Common poppy (Papaver rhoeas)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Iconic cornfield flower.", "wikipedia_title": "Common poppy"},
    {"title": "Meadow buttercup (Ranunculus acris)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Common in grasslands; mildly toxic (already covered under Ranunculus).", "wikipedia_title": "Meadow buttercup"},
    {"title": "Red campion (Silene dioica)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Common woodland edge flower.", "wikipedia_title": "Red campion"},
    {"title": "Common vetch (Vicia sativa)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Climbing plant with purple flowers.", "wikipedia_title": "Common vetch"},
    {"title": "Yarrow (Achillea millefolium)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Common on roadsides; medicinal.", "wikipedia_title": "Yarrow"},
    {"title": "Oxeye daisy (Leucanthemum vulgare)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Classic meadow daisy.", "wikipedia_title": "Oxeye daisy"},
    {"title": "Cowslip (Primula veris)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Spring flower, common in grassy places.", "wikipedia_title": "Cowslip"},
    {"title": "Primrose (Primula vulgaris)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Early spring flower, woodland.", "wikipedia_title": "Primrose"},
    {"title": "Musk mallow (Malva moschata)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Pink flower, common in dry places.", "wikipedia_title": "Musk mallow"},

    # --- Common garden plants (ornamentals, non-toxic) ---
    {"title": "Rose (Rosa spp.)", "category": "common_plant", "source_type": "wikipedia", "notes": "Most common garden shrub.", "wikipedia_title": "Rose"},
    {"title": "Tulip (Tulipa spp.)", "category": "common_plant", "source_type": "wikipedia", "notes": "Spring bulb, common.", "wikipedia_title": "Tulip"},
    {"title": "Lavender (Lavandula angustifolia)", "category": "common_plant", "source_type": "wikipedia", "notes": "Fragrant garden plant.", "wikipedia_title": "Lavender"},
    {"title": "Sunflower (Helianthus annuus)", "category": "common_plant", "source_type": "wikipedia", "notes": "Tall annual, common in gardens.", "wikipedia_title": "Sunflower"},
    {"title": "Petunia (Petunia spp.)", "category": "common_plant", "source_type": "wikipedia", "notes": "Popular bedding plant.", "wikipedia_title": "Petunia"},
    {"title": "Geranium (Pelargonium spp.)", "category": "common_plant", "source_type": "wikipedia", "notes": "Common pot plant.", "wikipedia_title": "Geranium"},
    {"title": "Fuchsia (Fuchsia magellanica)", "category": "common_plant", "source_type": "wikipedia", "notes": "Common garden shrub, also invasive in mild areas.", "wikipedia_title": "Fuchsia"},
    {"title": "Hydrangea (Hydrangea macrophylla)", "category": "common_plant", "source_type": "wikipedia", "notes": "Common garden shrub.", "wikipedia_title": "Hydrangea"},
    {"title": "Rhubarb (Rheum rhabarbarum)", "category": "common_plant", "source_type": "wikipedia", "notes": "Edible plant, but leaves toxic.", "wikipedia_title": "Rhubarb"},

    # --- Common trees ---
    {"title": "English oak (Quercus robur)", "category": "common_tree", "source_type": "wikipedia", "notes": "Iconic UK tree.", "wikipedia_title": "English oak"},
    {"title": "Sessile oak (Quercus petraea)", "category": "common_tree", "source_type": "wikipedia", "notes": "Common oak, similar.", "wikipedia_title": "Sessile oak"},
    {"title": "Sycamore (Acer pseudoplatanus)", "category": "common_tree", "source_type": "wikipedia", "notes": "Very common in UK.", "wikipedia_title": "Sycamore"},
    {"title": "Horse chestnut (Aesculus hippocastanum)", "category": "common_tree", "source_type": "wikipedia", "notes": "Conker tree; common.", "wikipedia_title": "Horse chestnut"},
    {"title": "Silver birch (Betula pendula)", "category": "common_tree", "source_type": "wikipedia", "notes": "Common pioneer tree.", "wikipedia_title": "Silver birch"},
    {"title": "Common ash (Fraxinus excelsior)", "category": "common_tree", "source_type": "wikipedia", "notes": "Widespread, but affected by ash dieback.", "wikipedia_title": "Common ash"},
    {"title": "Beech (Fagus sylvatica)", "category": "common_tree", "source_type": "wikipedia", "notes": "Common in southern UK.", "wikipedia_title": "Beech"},
    {"title": "Scots pine (Pinus sylvestris)", "category": "common_tree", "source_type": "wikipedia", "notes": "Native pine.", "wikipedia_title": "Scots pine"},
    {"title": "Corsican pine (Pinus nigra subsp. laricio)", "category": "common_tree", "source_type": "wikipedia", "notes": "Planted in forestry.", "wikipedia_title": "Corsican pine"},
    {"title": "Common lime (Tilia × europaea)", "category": "common_tree", "source_type": "wikipedia", "notes": "Often planted in parks.", "wikipedia_title": "Common lime"},
    {"title": "White willow (Salix alba)", "category": "common_tree", "source_type": "wikipedia", "notes": "Riverbank tree.", "wikipedia_title": "White willow"},
    {"title": "Crack willow (Salix fragilis)", "category": "common_tree", "source_type": "wikipedia", "notes": "Similar to white willow.", "wikipedia_title": "Crack willow"},
    {"title": "Alder (Alnus glutinosa)", "category": "common_tree", "source_type": "wikipedia", "notes": "Wetland tree.", "wikipedia_title": "Alder"},
    {"title": "Field maple (Acer campestre)", "category": "common_tree", "source_type": "wikipedia", "notes": "Native maple.", "wikipedia_title": "Field maple"},

    # --- Common mammals ---
    {"title": "Red fox (Vulpes vulpes)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Very common urban and rural.", "wikipedia_title": "Red fox"},
    {"title": "European hedgehog (Erinaceus europaeus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Common garden visitor, declining.", "wikipedia_title": "European hedgehog"},
    {"title": "Rabbit (Oryctolagus cuniculus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Abundant, often seen.", "wikipedia_title": "Rabbit"},
    {"title": "Brown hare (Lepus europaeus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Common in fields.", "wikipedia_title": "Brown hare"},
    {"title": "Roe deer (Capreolus capreolus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Common deer species.", "wikipedia_title": "Roe deer"},
    {"title": "Red deer (Cervus elaphus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Largest UK land mammal.", "wikipedia_title": "Red deer"},
    {"title": "Muntjac deer (Muntiacus reevesi)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Invasive but common; browses woodland understorey heavily.", "wikipedia_title": "Muntjac deer"},
    {"title": "Common shrew (Sorex araneus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Small mammal, common.", "wikipedia_title": "Common shrew"},
    {"title": "Pipistrelle bat (Pipistrellus pipistrellus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Most common UK bat.", "wikipedia_title": "Pipistrelle bat"},
    {"title": "Stoat (Mustela erminea)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Common predator.", "wikipedia_title": "Stoat"},
    {"title": "Weasel (Mustela nivalis)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Small mustelid.", "wikipedia_title": "Weasel"},

    # --- Additional common insects ---
    {"title": "Large white butterfly (Pieris brassicae)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common garden butterfly.", "wikipedia_title": "Large white butterfly"},
    {"title": "Small white butterfly (Pieris rapae)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common garden butterfly. wikipedia_title corrected to the binomial -- 'Small white' is a genuine Wikipedia DISAMBIGUATION page (shared with a pig breed and an African butterfly genus, Dixeia), which has no plain-text extract by design (VERIFIED via search this session: en.wikipedia.org/wiki/Small_white is the disambig page; en.wikipedia.org/wiki/Pieris_rapae is the real article).", "wikipedia_title": "Pieris rapae"},
    {"title": "Green-veined white (Pieris napi)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common.", "wikipedia_title": "Green-veined white"},
    {"title": "Common carder bee (Bombus pascuorum)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common bumblebee.", "wikipedia_title": "Common carder bee"},
    {"title": "Tree bumblebee (Bombus hypnorum)", "category": "common_insect", "source_type": "wikipedia", "notes": "Recent colonist, common.", "wikipedia_title": "Tree bumblebee"},
    {"title": "Common red soldier beetle (Rhagonycha fulva)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common garden beetle.", "wikipedia_title": "Common red soldier beetle"},
    {"title": "Harvestman (Opiliones)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common arachnid in gardens, not a spider.", "wikipedia_title": "Harvestman"},
    {"title": "Lacewing (Chrysopidae)", "category": "common_insect", "source_type": "wikipedia", "notes": "Green insect, common.", "wikipedia_title": "Lacewing"},
    {"title": "Common frog (Rana temporaria)", "category": "common_amphibian", "source_type": "wikipedia", "notes": "Common amphibian.", "wikipedia_title": "Common frog"},
    {"title": "Common toad (Bufo bufo)", "category": "common_amphibian", "source_type": "wikipedia", "notes": "Common.", "wikipedia_title": "Common toad"},
    {"title": "Smooth newt (Lissotriton vulgaris)", "category": "common_amphibian", "source_type": "wikipedia", "notes": "Common newt.", "wikipedia_title": "Smooth newt"},

    # --- Additional common fungi (non-toxic, edible or harmless) ---
    {"title": "Common puffball (Lycoperdon perlatum)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Edible when young.", "wikipedia_title": "Common puffball"},
    {"title": "Giant puffball (Calvatia gigantea)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Large edible fungus.", "wikipedia_title": "Giant puffball"},
    {"title": "Shaggy inkcap (Coprinus comatus)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Edible when young; same species as shaggy ink cap under fungi lookalikes.", "wikipedia_title": "Shaggy inkcap"},
    {"title": "Common stinkhorn (Phallus impudicus)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Not edible, but common and distinctive.", "wikipedia_title": "Common stinkhorn"},
    {"title": "Chicken of the woods (Laetiporus sulphureus)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Edible bracket fungus.", "wikipedia_title": "Chicken of the woods"},
    {"title": "Beefsteak fungus (Fistulina hepatica)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Edible, looks like raw meat.", "wikipedia_title": "Beefsteak fungus"},
    {"title": "Artist's bracket (Ganoderma applanatum)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Common tree fungus.", "wikipedia_title": "Artist's bracket"},
    {"title": "Birch polypore (Fomitopsis betulina)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Common on birch.", "wikipedia_title": "Birch polypore"},
    {"title": "Common earthball (Scleroderma citrinum)", "category": "common_fungus", "source_type": "wikipedia", "notes": "Poisonous lookalike for puffballs.", "wikipedia_title": "Common earthball"},

    # --- Additional common birds not yet listed ---
    {"title": "Mallard (Anas platyrhynchos)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common duck.", "wikipedia_title": "Mallard"},
    {"title": "Moorhen (Gallinula chloropus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common waterbird.", "wikipedia_title": "Moorhen"},
    {"title": "Coot (Fulica atra)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common waterbird. wikipedia_title corrected -- plain 'Coot' resolves to the genus-level article (Fulica, 10+ species worldwide), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Eurasian_coot is Fulica atra specifically).", "wikipedia_title": "Eurasian coot"},
    {"title": "Herring gull (Larus argentatus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common coastal gull.", "wikipedia_title": "Herring gull"},
    {"title": "Black-headed gull (Chroicocephalus ridibundus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common inland gull.", "wikipedia_title": "Black-headed gull"},
    {"title": "Common tern (Sterna hirundo)", "category": "common_bird", "source_type": "wikipedia", "notes": "Summer visitor, common.", "wikipedia_title": "Common tern"},
    {"title": "Great spotted woodpecker (Dendrocopos major)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common garden woodpecker.", "wikipedia_title": "Great spotted woodpecker"},
    {"title": "Green woodpecker (Picus viridis)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common on lawns.", "wikipedia_title": "Green woodpecker"},
    {"title": "Nuthatch (Sitta europaea)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common woodland bird. wikipedia_title corrected -- plain 'Nuthatch' resolves to the genus-level article (Sitta, 25+ species worldwide), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Eurasian_nuthatch is Sitta europaea specifically).", "wikipedia_title": "Eurasian nuthatch"},
    {"title": "Treecreeper (Certhia familiaris)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common. wikipedia_title corrected -- plain 'Treecreeper' resolves to the family-level article (Certhiidae, 9 species), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Eurasian_treecreeper is Certhia familiaris specifically).", "wikipedia_title": "Eurasian treecreeper"},
    {"title": "Coal tit (Periparus ater)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common garden bird.", "wikipedia_title": "Coal tit"},
    {"title": "Long-tailed tit (Aegithalos caudatus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common and distinctive.", "wikipedia_title": "Long-tailed tit"},
    {"title": "Bullfinch (Pyrrhula pyrrhula)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common garden bird. wikipedia_title corrected -- plain 'Bullfinch' resolves to the genus-level article (Pyrrhula, 8 species across Europe/Asia), not the specific UK species (VERIFIED via search this session: en.wikipedia.org/wiki/Eurasian_bullfinch is Pyrrhula pyrrhula specifically).", "wikipedia_title": "Eurasian bullfinch"},
    {"title": "Linnet (Linaria cannabina)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common farmland bird. wikipedia_title corrected -- plain 'Linnet' is a genuine Wikipedia DISAMBIGUATION page, which has no plain-text extract by design (VERIFIED via search this session: en.wikipedia.org/wiki/Common_linnet is Linaria cannabina specifically).", "wikipedia_title": "Common linnet"},
    {"title": "Skylark (Alauda arvensis)", "category": "common_bird", "source_type": "wikipedia", "notes": "Iconic songbird. wikipedia_title corrected as a precaution -- the genus Alauda contains 4 species, so plain 'Skylark' carries the same disambiguation/redirect risk confirmed elsewhere in this manifest (VERIFIED via search this session: en.wikipedia.org/wiki/Eurasian_skylark is Alauda arvensis specifically).", "wikipedia_title": "Eurasian skylark"},

    # --- Taxonomic / conceptual backbone pages (Wikipedia) ---
    {"title": "Species", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Core concept page.", "wikipedia_title": "Species"},
    {"title": "Binomial nomenclature", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "", "wikipedia_title": "Binomial nomenclature"},
    {"title": "Taxonomic rank", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "", "wikipedia_title": "Taxonomic rank"},
    {"title": "Cryptic species", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Directly relevant to lookalike-warning stretch goal.", "wikipedia_title": "Cryptic species"},
    {"title": "Invasive species", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "", "wikipedia_title": "Invasive species"},
    {"title": "IUCN Red List", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "", "wikipedia_title": "IUCN Red List"},
    {"title": "Wildlife and Countryside Act 1981", "category": "legal_concept", "source_type": "wikipedia", "notes": "Primary UK protected-species legislation — background context, not a substitute for the primary legal text.", "wikipedia_title": "Wildlife and Countryside Act 1981"},

    # --- Expansion: more invasive/protected/toxic species and lookalike pairs ---
    {"title": "Quagga mussel (Dreissena rostriformis bugensis)", "category": "invasive_invertebrate", "source_type": "wikipedia", "notes": "Recently established UK freshwater invasive.", "wikipedia_title": "Quagga mussel"},
    {"title": "Killer shrimp (Dikerogammarus villosus)", "category": "invasive_invertebrate", "source_type": "wikipedia", "notes": "Aquatic invasive, displaces native invertebrates.", "wikipedia_title": "Killer shrimp"},
    {"title": "Slender speedwell (Veronica filiformis)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Common invasive lawn plant, low ecological concern but frequently misidentified.", "wikipedia_title": "Slender speedwell"},
    {"title": "Three-cornered leek (Allium triquetrum)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Invasive, resembles wild garlic -- relevant lookalike pairing.", "wikipedia_title": "Three-cornered leek"},
    {"title": "Winter heliotrope (Petasites fragrans)", "category": "invasive_plant", "source_type": "wikipedia", "notes": "Invasive groundcover, spreads along roadsides and riverbanks.", "wikipedia_title": "Winter heliotrope"},
    {"title": "Bee orchid (Ophrys apifera)", "category": "common_wildflower", "source_type": "wikipedia", "notes": "Do not label as legally protected from this manifest alone; conservation status and any local/site protection should be verified separately. wikipedia_title corrected to the binomial -- 'Bee orchid' is a genuine Wikipedia DISAMBIGUATION page (shared with three other orchid species from India, Australia, and Puerto Rico that also go by 'bee orchid'), which has no plain-text extract by design (VERIFIED via search this session: en.wikipedia.org/wiki/Bee_orchid is the disambig page; en.wikipedia.org/wiki/Ophrys_apifera is the real UK-species article).", "wikipedia_title": "Ophrys apifera"},
    {"title": "Marsh helleborine (Epipactis palustris)", "category": "notable_plant", "source_type": "wikipedia", "notes": "Wetland orchid; do not infer statutory Schedule 8 protection without checking current legislation.", "wikipedia_title": "Marsh helleborine"},
    {"title": "Cuckoopint (Arum maculatum)", "category": "toxic_plant", "source_type": "wikipedia", "notes": "All parts toxic, berries attractive to children.", "wikipedia_title": "Cuckoopint"},
    {"title": "Panther cap lookalike: The Blusher (Amanita rubescens)", "category": "edible_fungus_lookalike", "source_type": "wikipedia", "notes": "Edible when cooked, confused with toxic panther cap.", "wikipedia_title": "Amanita rubescens"},
    {"title": "Livid pinkgill (Entoloma sinuatum)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Toxic, resembles edible Clitocybe species.", "wikipedia_title": "Livid pinkgill"},
    {"title": "Common ink cap (Coprinopsis atramentaria)", "category": "toxic_fungus", "source_type": "wikipedia", "notes": "Toxic in combination with alcohol -- important safety-behaviour example.", "wikipedia_title": "Common ink cap"},
    {"title": "Ivy bee (Colletes hederae)", "category": "common_insect", "source_type": "wikipedia", "notes": "Recently colonised UK, often mistaken for wasp due to size/colour.", "wikipedia_title": "Ivy bee"},
    {"title": "Hornet mimic hoverfly (Volucella zonaria)", "category": "native_animal_lookalike", "source_type": "wikipedia", "notes": "Harmless fly mimicking hornets -- key disambiguation entry. wikipedia_title corrected to the scientific name; the common name has no dedicated page.", "wikipedia_title": "Volucella zonaria"},
    {"title": "Stag beetle (Lucanus cervus)", "category": "conservation_priority_animal", "source_type": "wikipedia", "notes": "UK's largest beetle and a conservation-priority species; do not equate conservation priority with blanket statutory protection.", "wikipedia_title": "Stag beetle"},
    {"title": "Slow worm (Anguis fragilis)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Legless lizard, frequently misidentified as a snake; partially protected.", "wikipedia_title": "Slow worm"},
    # --- Expansion 2: more real species, contrast pairs, and conservation-priority birds/insects ---
    {"title": "Cinnabar moth (Tyria jacobaeae)", "category": "common_insect", "source_type": "wikipedia", "notes": "Caterpillars feed on ragwort; pairs with the ragwort/common ragwort entries above.", "wikipedia_title": "Cinnabar moth"},
    {"title": "Painted lady (Vanessa cardui)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common migratory butterfly baseline.", "wikipedia_title": "Painted lady"},
    {"title": "Speckled wood (Pararge aegeria)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common woodland butterfly baseline.", "wikipedia_title": "Pararge aegeria"},
    {"title": "Orange tip (Anthocharis cardamines)", "category": "common_insect", "source_type": "wikipedia", "notes": "Common spring butterfly baseline.", "wikipedia_title": "Orange tip"},
    {"title": "European nightjar (Caprimulgus europaeus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected, ground-nesting heathland bird.", "wikipedia_title": "European nightjar"},
    {"title": "Hen harrier (Circus cyaneus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected; high-profile UK raptor persecution/conservation case.", "wikipedia_title": "Hen harrier"},
    {"title": "Willow tit (Poecile montanus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "One of the UK's most rapidly declining birds.", "wikipedia_title": "Willow tit"},
    {"title": "European turtle dove (Streptopelia turtur)", "category": "protected_animal", "source_type": "wikipedia", "notes": "UK's fastest-declining bird species; high conservation priority.", "wikipedia_title": "European turtle dove"},
    {"title": "White-clawed crayfish (Austropotamobius pallipes)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Native crayfish, displaced by invasive signal crayfish above -- key contrast pair.", "wikipedia_title": "White-clawed crayfish"},
    {"title": "Natterer's bat (Myotis nattereri)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Additional UK bat species; all UK bats are protected.", "wikipedia_title": "Natterer's bat"},
    {"title": "Marsh fritillary (Euphydryas aurinia)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Protected butterfly, EU Habitats Directive Annex II species.", "wikipedia_title": "Marsh fritillary"},
    {"title": "Large blue (Phengaris arion)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Extinct in UK 1979, reintroduced -- iconic conservation success story, protected.", "wikipedia_title": "Large blue"},
    {"title": "American bullfrog (Lithobates catesbeianus)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Non-native to UK; may be invasive where introduced; large frog, can outcompete native species.", "wikipedia_title": "American bullfrog"},
    {"title": "Marsh frog (Pelophylax ridibundus)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Introduced to UK; invasive, often confused with native pool frog.", "wikipedia_title": "Marsh frog"},
    {"title": "Green frog (Lithobates clamitans)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "North American species; not native to UK; potential invasive if introduced.", "wikipedia_title": "Lithobates clamitans"},
    {"title": "Ligurian emperor (Saturnia pavoniella)", "category": "common_insect", "source_type": "wikipedia", "notes": "European moth, similar to small emperor moth (Saturnia pavonia) but not native to UK.", "wikipedia_title": "Saturnia pavoniella"},
    {"title": "Dog (Canis familiaris)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Domestic species; not a wild animal, but included for pet‑related toxicity references.", "wikipedia_title": "Dog"},
    {"title": "Cat (Felis catus)", "category": "common_mammal", "source_type": "wikipedia", "notes": "Domestic species; not a wild animal, but included for pet‑related toxicity references.", "wikipedia_title": "Cat"},
    {"title": "Dog breed", "category": "pet_breed", "source_type": "wikipedia", "notes": "General page on dog breeds; useful for breed‑level identification queries.", "wikipedia_title": "Dog breed"},
    {"title": "Cat breed", "category": "pet_breed", "source_type": "wikipedia", "notes": "General page on cat breeds; useful for breed‑level identification queries.", "wikipedia_title": "Cat breed"},
    {"title": "List of dog breeds", "category": "pet_breed", "source_type": "wikipedia", "notes": "Comprehensive list, complements the main dog breed page.", "wikipedia_title": "List of dog breeds"},
    {"title": "List of cat breeds", "category": "pet_breed", "source_type": "wikipedia", "notes": "Comprehensive list, complements the main cat breed page.", "wikipedia_title": "List of cat breeds"},
    {"title": "Dog health", "category": "pet_health", "source_type": "wikipedia", "notes": "Covers diseases, nutrition, and common health issues in dogs.", "wikipedia_title": "Dog health"},
    {"title": "Cat health", "category": "pet_health", "source_type": "wikipedia", "notes": "Covers diseases, nutrition, and common health issues in cats.", "wikipedia_title": "Cat health"},
    {"title": "Dog behavior", "category": "animal_behavior", "source_type": "wikipedia", "notes": "Overview of canine behaviour, communication and social structure.", "wikipedia_title": "Dog behavior"},
    {"title": "Cat behavior", "category": "animal_behavior", "source_type": "wikipedia", "notes": "Overview of feline behaviour, communication and social structure.", "wikipedia_title": "Cat behavior"},
    {"title": "Poisoning in dogs", "category": "toxic_status", "source_type": "wikipedia", "notes": "Directly relevant to the safety‑critical pet‑toxicity queries (e.g., 'Is this plant toxic to my dog?').", "wikipedia_title": "Poisoning in dogs"},
    {"title": "Poisoning in cats", "category": "toxic_status", "source_type": "wikipedia", "notes": "Directly relevant to the safety‑critical pet‑toxicity queries (e.g., 'Is this plant toxic to my cat?').", "wikipedia_title": "Poisoning in cats"},
        # --- Additional UK birds (waders, raptors, seabirds, farmland, non-native) ---
    {"title": "Common buzzard (Buteo buteo)", "category": "common_bird", "source_type": "wikipedia", "notes": "Now the UK's most widespread raptor; absent from the previous list.", "wikipedia_title": "Common buzzard"},
    {"title": "Osprey (Pandion haliaetus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected; notable conservation success story after reintroduction.", "wikipedia_title": "Osprey"},
    {"title": "Red kite (Milvus milvus)", "category": "protected_animal", "source_type": "wikipedia", "notes": "Schedule 1 protected; iconic UK reintroduction success, now common in many areas.", "wikipedia_title": "Red kite"},
    {"title": "Eurasian curlew (Numenius arquata)", "category": "common_bird", "source_type": "wikipedia", "notes": "UK's largest wader; red‑listed (high conservation concern).", "wikipedia_title": "Eurasian curlew"},
    {"title": "Northern lapwing (Vanellus vanellus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Red‑listed farmland wader; known for its distinctive 'pee-wit' call.", "wikipedia_title": "Northern lapwing"},
    {"title": "Common snipe (Gallinago gallinago)", "category": "common_bird", "source_type": "wikipedia", "notes": "Widespread wader, often flushed from wetlands.", "wikipedia_title": "Common snipe"},
    {"title": "Eurasian oystercatcher (Haematopus ostralegus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common coastal wader with a bright orange bill.", "wikipedia_title": "Eurasian oystercatcher"},
    {"title": "Atlantic puffin (Fratercula arctica)", "category": "common_bird", "source_type": "wikipedia", "notes": "Iconic UK seabird; now vulnerable globally.", "wikipedia_title": "Atlantic puffin"},
    {"title": "Northern gannet (Morus bassanus)", "category": "common_bird", "source_type": "wikipedia", "notes": "UK's largest seabird; spectacular plunge‑diver.", "wikipedia_title": "Northern gannet"},
    {"title": "Common guillemot (Uria aalge)", "category": "common_bird", "source_type": "wikipedia", "notes": "Abundant seabird on UK cliffs.", "wikipedia_title": "Common guillemot"},
    {"title": "Razorbill (Alca torda)", "category": "common_bird", "source_type": "wikipedia", "notes": "Seabird similar to guillemot, but with a deeper bill.", "wikipedia_title": "Razorbill"},
    {"title": "Black‑legged kittiwake (Rissa tridactyla)", "category": "common_bird", "source_type": "wikipedia", "notes": "The only UK gull that nests on sheer cliff ledges.", "wikipedia_title": "Black-legged kittiwake"},
    {"title": "Grey partridge (Perdix perdix)", "category": "common_bird", "source_type": "wikipedia", "notes": "Red‑listed farmland bird; severe population decline.", "wikipedia_title": "Grey partridge"},
    {"title": "Red grouse (Lagopus lagopus scotica)", "category": "common_bird", "source_type": "wikipedia", "notes": "Subspecies endemic to Britain; managed as a gamebird.", "wikipedia_title": "Red grouse"},
    {"title": "Ring‑necked parakeet (Psittacula krameri)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Naturalised non‑native; now widespread in southeast England. Good contrast case for 'what's native vs. invasive?'.", "wikipedia_title": "Ring-necked parakeet"},
    {"title": "Yellowhammer (Emberiza citrinella)", "category": "common_bird", "source_type": "wikipedia", "notes": "Farmland bunting; red‑listed, known for its 'little bit of bread and no cheese' song.", "wikipedia_title": "Yellowhammer"},
    {"title": "Goldcrest (Regulus regulus)", "category": "common_bird", "source_type": "wikipedia", "notes": "UK's smallest bird; common in conifers.", "wikipedia_title": "Goldcrest"},
    {"title": "Eurasian blackcap (Sylvia atricapilla)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common warbler; some UK populations now overwinter.", "wikipedia_title": "Eurasian blackcap"},
    {"title": "Great crested grebe (Podiceps cristatus)", "category": "common_bird", "source_type": "wikipedia", "notes": "Elegant waterbird with a beautiful courtship display.", "wikipedia_title": "Great crested grebe"},
    {"title": "Little egret (Egretta garzetta)", "category": "common_bird", "source_type": "wikipedia", "notes": "Colonised the UK naturally in the 1990s; now common on wetlands.", "wikipedia_title": "Little egret"},
    {"title": "Pied wagtail (Motacilla alba yarrellii)", "category": "common_bird", "source_type": "wikipedia", "notes": "Common garden and car‑park bird; distinctive bobble. British subspecies (yarrellii) is black‑backed.", "wikipedia_title": "White wagtail"},
    {"title": "Eurasian hobby (Falco subbuteo)", "category": "common_bird", "source_type": "wikipedia", "notes": "Summer migrant; a small, fast falcon known for catching dragonflies and swallows in flight.", "wikipedia_title": "Eurasian hobby"},
    # --- Frogs & Moths (already in SPECIES_WIKI; provided here as a ready-to-use block) ---
    {"title": "American bullfrog (Lithobates catesbeianus)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Non-native to UK; may be invasive where introduced; large frog, can outcompete native species.", "wikipedia_title": "American bullfrog"},
    {"title": "Marsh frog (Pelophylax ridibundus)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "Introduced to UK; invasive, often confused with native pool frog.", "wikipedia_title": "Marsh frog"},
    {"title": "Green frog (Lithobates clamitans)", "category": "invasive_animal", "source_type": "wikipedia", "notes": "North American species; not native to UK; potential invasive if introduced.", "wikipedia_title": "Lithobates clamitans"},
    {"title": "Emperor moth (Saturnia pavonia)", "category": "common_insect", "source_type": "wikipedia", "notes": "VERIFIED via search this session. Britain's only native Saturniidae; common on moorland/heathland, non-toxic, no protected/invasive status -- a good baseline non-flagged ID case. Matches the misidentification example already used in the app's own test transcript.", "wikipedia_title": "Saturnia pavonia"},
    {"title": "Ligurian emperor (Saturnia pavoniella)", "category": "common_insect", "source_type": "wikipedia", "notes": "European moth, similar to small emperor moth (Saturnia pavonia) but not native to UK.", "wikipedia_title": "Saturnia pavoniella"},
    {"title": "Small emperor moth (Saturnia pavonia)", "category": "common_insect", "source_type": "wikipedia", "notes": "VERIFIED via search this session. Britain's only native Saturniidae; common on moorland/heathland, non-toxic, no protected/invasive status -- a good baseline non-flagged ID case. Matches the misidentification example already used in the app's own test transcript.", "wikipedia_title": "Saturnia pavonia"},
        # --- Non-native trees (ornamental / naturalised) ---
    {"title": "Swedish whitebeam (Scandosorbus intermedia)", "category": "common_tree", "source_type": "wikipedia", "notes": "Non‑native ornamental tree from Scandinavia, widely planted in UK parks and gardens; occasionally naturalises. Not legally protected and not listed as invasive in the UK. The genus was formerly included in Sorbus; current accepted name is Scandosorbus intermedia.", "wikipedia_title": "Scandosorbus intermedia"},
    # --- Swedish whitebeam / Sorbus Wikipedia sources ---
    {"title": "Scandosorbus intermedia (Swedish whitebeam)", "category": "common_tree", "source_type": "wikipedia", "notes": "Primary species account. Current name Scandosorbus intermedia; formerly widely treated as Sorbus intermedia.", "wikipedia_title": "Scandosorbus intermedia"},

    {"title": "Sorbus", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Useful genus-level background for the former classification of Swedish whitebeam and related whitebeams.", "wikipedia_title": "Sorbus"},

    {"title": "Rosaceae", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Family-level context for Swedish whitebeam and related genera.", "wikipedia_title": "Rosaceae"},

    {"title": "Apomixis", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Relevant to the reproductive biology of Scandosorbus intermedia, which is described as an apomictic polyploid.", "wikipedia_title": "Apomixis"},

    {"title": "Polyploidy", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Useful conceptual background for the tetraploid biology of Swedish whitebeam and other hybridogenous Sorbus taxa.", "wikipedia_title": "Polyploidy"},

    {"title": "Hybridization", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Relevant to the hybrid origin and relationships of Swedish whitebeam within the Sorbus complex.", "wikipedia_title": "Hybridization"},

    {"title": "List of Rosaceae of Great Britain and Ireland", "category": "taxonomy_concept", "source_type": "wikipedia", "notes": "Useful UK-specific cross-reference showing Swedish whitebeam under Scandosorbus and related British Sorbus taxa.", "wikipedia_title": "List of Rosaceae of Great Britain and Ireland"},
]

# ---------------------------------------------------------------------------
# 2. REGIONAL STATUS SOURCES (UK)
#    URL fields have been set with plausible government/NGO web addresses;
#    verify each before use.
# ---------------------------------------------------------------------------

REGIONAL_STATUS_SOURCES = [
    {"title": "GB Non-native Species Secretariat — species alerts and lists", "category": "invasive_status", "source_type": "gov", "notes": "Primary source for GB invasive species status; ingest the actual current list pages, not this description. URL corrected and re-verified via search (old /nonnativespecies/alerts/ path was stale).", "url": "https://www.nonnativespecies.org/non-native-species/species-alerts"},
    {"title": "Wildlife and Countryside Act 1981, Schedule 5 (protected animals)", "category": "protected_status", "source_type": "gov", "notes": "Legal text/schedule listing.", "url": "https://www.legislation.gov.uk/ukpga/1981/69/schedule/5"},
    {"title": "Wildlife and Countryside Act 1981, Schedule 8 (protected plants)", "category": "protected_status", "source_type": "gov", "notes": "Legal text/schedule listing.", "url": "https://www.legislation.gov.uk/ukpga/1981/69/schedule/8"},
    {"title": "Wildlife and Countryside Act 1981, Schedule 9 (invasive non-natives release restrictions)", "category": "invasive_status", "source_type": "gov", "notes": "", "url": "https://www.legislation.gov.uk/ukpga/1981/69/schedule/9"},
    {"title": "JNCC — UK protected species pages", "category": "protected_status", "source_type": "gov", "notes": "Joint Nature Conservation Committee reference material.", "url": "https://jncc.gov.uk/our-work/uk-protected-species/"},
    {"title": "Royal Horticultural Society — Poisonous plants list", "category": "toxic_status", "source_type": "org", "notes": "Named explicitly in the project brief.", "url": "https://www.rhs.org.uk/advice/profile?pid=484"},
    {"title": "GOV.UK — Invasive non-native (alien) plant species: rules in England", "category": "invasive_status", "source_type": "gov", "notes": "Legal duties around Japanese knotweed etc.", "url": "https://www.gov.uk/guidance/invasive-non-native-alien-plant-species-rules-in-england"},
    {"title": "GOV.UK — Prevent the spread of invasive non-native plants", "category": "invasive_status", "source_type": "gov", "notes": "", "url": "https://www.gov.uk/guidance/prevent-the-spread-of-invasive-non-native-plants"},
    {"title": "Natural England — Species licensing guidance", "category": "protected_status", "source_type": "gov", "notes": "Licensing exceptions for protected species handling.", "url": "https://www.gov.uk/guidance/species-licensing"},
    {"title": "Environment Agency — Invasive non-native species guidance for land managers", "category": "invasive_status", "source_type": "gov", "notes": "", "url": "https://www.gov.uk/guidance/invasive-non-native-species-guidance-for-land-managers"},
    {"title": "British Mycological Society — Guidance on poisonous fungi", "category": "toxic_status", "source_type": "org", "notes": "Complements RHS list for fungi specifically.", "url": "https://www.britmycolsoc.org.uk/guidance/poisonous-fungi"},
    {"title": "UK Health Security Agency — Poisonous plants and fungi advice", "category": "toxic_status", "source_type": "gov", "notes": "Public health angle for the safety-warning behaviour.", "url": "https://www.gov.uk/government/collections/poisonous-plants-and-fungi"},
    {"title": "GBIF — UK species occurrence records", "category": "distribution_data", "source_type": "gbif", "notes": "Confirms which species are realistically present in the user's stated region.", "url": "https://www.gbif.org/occurrence/search?country=GB"},
    {"title": "iRecord / National Biodiversity Network (NBN) Atlas — UK species records", "category": "distribution_data", "source_type": "org", "notes": "Alternative/complementary UK occurrence data source.", "url": "https://nbnatlas.org/"},
    {"title": "The Invasive Alien Species (Enforcement and Permitting) Order 2019", "category": "invasive_status", "source_type": "gov", "notes": "UK legislation implementing EU regulation on IAS.", "url": "https://www.legislation.gov.uk/uksi/2019/1172/contents/made"},
    {"title": "Protection of Badgers Act 1992", "category": "protected_status", "source_type": "gov", "notes": "Legal protection for badgers.", "url": "https://www.legislation.gov.uk/ukpga/1992/51/contents"},
    {"title": "Countryside and Rights of Way Act 2000 (CROW) — protection of SSSIs and species", "category": "protected_status", "source_type": "gov", "notes": "Reinforces species protection in designated sites.", "url": "https://www.legislation.gov.uk/ukpga/2000/37/contents"},
    {"title": "UK Biodiversity Action Plan (BAP) priority species lists", "category": "protected_status", "source_type": "gov", "notes": "Lists of species of conservation concern; may overlap with protected species.", "url": "https://jncc.gov.uk/our-work/uk-bap-priority-species/"},
    {"title": "Local Environmental Records Centres (LERC) — regional rarity status", "category": "distribution_data", "source_type": "org", "notes": "Useful for local rarity, but need to contact each centre.", "url": None},
    {"title": "Plantlife — UK wildflower species guides and conservation status", "category": "protected_status", "source_type": "org", "notes": "NGO with conservation status for plants.", "url": "https://www.plantlife.org.uk/"},
    {"title": "Butterfly Conservation — UK butterfly species status and conservation", "category": "protected_status", "source_type": "org", "notes": "NGO with red lists for butterflies.", "url": "https://butterfly-conservation.org/"},

    # --- Expansion: further UK/devolved-nation gov and NGO sources ---
    {"title": "Forestry Commission — Tree pests and diseases guidance", "category": "invasive_status", "source_type": "gov", "notes": "Covers oak processionary moth and other tree health threats.", "url": "https://www.forestresearch.gov.uk/tools-and-resources/pest-and-disease-resources/"},
    {"title": "GB Invasive Non-native Species Strategy", "category": "invasive_status", "source_type": "gov", "notes": "DEFRA/GB-wide strategy document; ingest current published version.", "url": "https://www.nonnativespecies.org/assets/Uploads/gb-nnss-strategy-2023.pdf"},
    {"title": "NatureScot — Species licensing (Scotland)", "category": "protected_status", "source_type": "gov", "notes": "Scotland-specific equivalent to Natural England licensing guidance.", "url": "https://www.nature.scot/professional-advice/safeguarding-protected-areas-and-species/protected-species/species-licensing"},
    {"title": "Natural Resources Wales — Invasive non-native species", "category": "invasive_status", "source_type": "gov", "notes": "Wales-specific invasive species guidance.", "url": "https://naturalresources.wales/guidance-and-advice/environmental-topics/wildlife-and-biodiversity/invasive-non-native-species/"},
    {"title": "Northern Ireland Environment Agency — Protected species", "category": "protected_status", "source_type": "gov", "notes": "Northern Ireland-specific protected species guidance.", "url": "https://www.daera-ni.gov.uk/articles/protected-species"},
    {"title": "GB Non-native Species Secretariat — Horizon scanning and risk assessment", "category": "invasive_status", "source_type": "gov", "notes": "Risk assessments for species not yet established in GB.", "url": "https://www.nonnativespecies.org/non-native-species/horizon-scanning-tool/"},
    {"title": "UK Health Security Agency — Asian hornet public health advice", "category": "toxic_status", "source_type": "gov", "notes": "Sting/allergic-reaction advice, complements the Asian hornet species entry.", "url": None},
    {"title": "DEFRA — Code of Practice on How to Prevent the Spread of Ragwort", "category": "toxic_status", "source_type": "gov", "notes": "Statutory guidance under the Ragwort Control Act 2003.", "url": "https://www.gov.uk/government/publications/ragwort-control"},
    {"title": "Marine Management Organisation — Non-native marine species", "category": "invasive_status", "source_type": "gov", "notes": "Covers marine invasives not addressed by terrestrial-focused sources above.", "url": None},
    {"title": "Wildlife Trusts — UK species and habitat conservation status", "category": "protected_status", "source_type": "org", "notes": "Federation of 46 local Wildlife Trusts; broad species/habitat conservation coverage.", "url": "https://www.wildlifetrusts.org/"},

    # --- Expansion 3: verified this session via web search ---
    {"title": "National Poisons Information Service (NPIS) / TOXBASE", "category": "toxic_status", "source_type": "gov", "notes": "VERIFIED via search. UK's clinical poisons information service, commissioned by UKHSA; TOXBASE has mushroom/plant identification photos used by UK clinicians for poisoning cases. TOXBASE itself (toxbase.org) requires NHS/registered-department login -- npis.org is the public-facing site; ingest what's publicly accessible, and treat TOXBASE's clinical database as reference-only, not something to scrape.", "url": "https://www.npis.org/"},

    # --- Pet (dog/cat) toxicity reference sources -- these are what the
    #     regional-status tool would check a species/plant against for the
    #     brief's worked example ("Is this spider dangerous to my dog?").
    #     NOTE: generic "Dog"/"Cat" SPECIES_WIKI entries were added separately
    #     above (see the invasive_animal/common_mammal block) for pet-related
    #     toxicity references -- this comment previously said the opposite
    #     (that they were deliberately excluded); that was stale and wrong,
    #     left over from before those two entries were added. Corrected here
    #     so the comment matches what's actually in SPECIES_WIKI.
    {"title": "PDSA — Poisonous plants (dogs, cats, other pets)", "category": "toxic_status", "source_type": "org", "notes": "VERIFIED via search this session. UK veterinary charity; practical species-by-species list of plants poisonous to pets, complements the RHS/human-focused poisonous plants list already above.", "url": "https://www.pdsa.org.uk/pet-help-and-advice/looking-after-your-pet/all-pets/poisonous-plants"},
    {"title": "Animal PoisonLine (UK) — Veterinary Poisons Information Service", "category": "toxic_status", "source_type": "org", "notes": "VERIFIED via search this session (cross-checked against vpisglobal.com and pdsa.org.uk, both point to this as the real public number/site). UK's only 24-hour public pet-poisoning triage line (01202 509000, charges apply); relevant if the app's safety behaviour ever needs to point a user toward acting on a suspected pet poisoning, analogous to how NPIS/TOXBASE covers the human case above.", "url": "https://www.animalpoisonline.co.uk/"},
]

# ---------------------------------------------------------------------------
# 3. FIELD GUIDE / NATURAL HISTORY BOOKS
#    ISBNs left as None; you can look up the exact editions via Open Library.
# ---------------------------------------------------------------------------

BOOKS = [
    {"title": "On the Origin of Species", "category": "book_public_domain", "source_type": "open_library", "notes": "Charles Darwin, 1859.", "isbn": None},
    {"title": "The Voyage of the Beagle", "category": "book_public_domain", "source_type": "open_library", "notes": "Charles Darwin.", "isbn": None},
    {"title": "The Malay Archipelago", "category": "book_public_domain", "source_type": "open_library", "notes": "Alfred Russel Wallace.", "isbn": None},
    {"title": "The Natural History of Selborne", "category": "book_public_domain", "source_type": "open_library", "notes": "Gilbert White — foundational UK natural history text.", "isbn": None},
    {"title": "English Botany", "category": "book_public_domain", "source_type": "open_library", "notes": "James Sowerby and James Edward Smith, 1790-1814. Replaces prior vague title -- Flora Britannica (1996, Richard Mabey) is NOT public domain, was a data error.", "isbn": None},
    {"title": "The Insect World of J. Henri Fabre", "category": "book_public_domain", "source_type": "open_library", "notes": "English translation of Souvenirs Entomologiques.", "isbn": None},
    {"title": "Outlines of British Fungology", "category": "book_public_domain", "source_type": "open_library", "notes": "Worthington G. Smith, L. Reeve & Co, 1870 (supplement 1891).", "isbn": None},
    {"title": "A Manual of British Botany", "category": "book_public_domain", "source_type": "open_library", "notes": "Charles Cardale Babington, 19th-century UK flora reference.", "isbn": None},
    {"title": "The Birds of Great Britain", "category": "book_public_domain", "source_type": "open_library", "notes": "John Gould, 5 volumes, 1862-1873.", "isbn": None},
    {"title": "Curtis's Botanical Magazine", "category": "book_public_domain", "source_type": "open_library", "notes": "Long-running illustrated botanical periodical; early volumes are PD. Parenthetical clarifier removed -- it was breaking Open Library search matching.", "isbn": None},
    {"title": "British Entomology (John Curtis)", "category": "book_public_domain", "source_type": "open_library", "notes": "Illustrated early UK insect reference.", "isbn": None},
    {"title": "A History of British Birds (William Yarrell)", "category": "book_public_domain", "source_type": "open_library", "notes": "", "isbn": None},
    {"title": "Wild Flowers", "category": "book_public_domain", "source_type": "open_library", "notes": "Anne Pratt, 1852-53, 2 volumes.", "isbn": None},
    {"title": "The Fungus Flora of Yorkshire", "category": "book_public_domain", "source_type": "open_library", "notes": "George Massee and Charles Crossland, 1905; full text on Internet Archive/BHL.", "isbn": None},
    {"title": "The Flowering Plants of Great Britain", "category": "book_public_domain", "source_type": "open_library", "notes": "Anne Pratt, 6 volumes, 1855-1873 (also known as The Flowering Plants, Grasses, Sedges, and Ferns of Great Britain). Replaces unresolvable author placeholder.", "isbn": None},
    {"title": "The Ferns of Great Britain and Ireland", "category": "book_public_domain", "source_type": "open_library", "notes": "Thomas Moore, edited by John Lindley, 1855. Corrected author attribution (was wrongly credited to J. E. Smith).", "isbn": None},
    {"title": "British Butterflies", "category": "book_public_domain", "source_type": "open_library", "notes": "W. S. Coleman, Routledge, 1860. Corrected title (was invented as A History of British Butterflies).", "isbn": None},
    {"title": "British Birds", "category": "book_public_domain", "source_type": "open_library", "notes": "W. H. Hudson, 1895. Replaces unresolvable various-authors placeholder.", "isbn": None},
    {"title": "Flora Scotica (John Lightfoot)", "category": "book_public_domain", "source_type": "open_library", "notes": "Scottish flora, 18th-century.", "isbn": None},
    {"title": "The British Herbal (John Hill)", "category": "book_public_domain", "source_type": "open_library", "notes": "18th-century medicinal plant guide.", "isbn": None},
    {"title": "Guide to Sowerby's Models of British Fungi", "category": "book_public_domain", "source_type": "open_library", "notes": "Worthington George Smith, British Museum (Natural History), 1898. Distinct real work from Outlines of British Fungology above.", "isbn": None},
    {"title": "Collins Bird Guide", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Standard UK field ID reference (2nd edition, paperback, Svensson/Mullarney/Zetterstrom/Grant, HarperCollins 2010); do not ingest full text without rights.", "isbn": "9780007268146"},
    {"title": "Collins Complete Guide to British Wildlife", "category": "book_copyrighted", "source_type": "reference_only", "notes": "", "isbn": None},
    {"title": "Roger Phillips — Mushrooms", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Widely used UK fungi field guide (2nd revised edition, Pan Macmillan 2006).", "isbn": "9780330442374"},
    {"title": "Francis Rose — The Wild Flower Key", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Standard UK botanical ID key (revised edition with Clare OReilly, Warne 2006).", "isbn": "9780723251750"},
    {"title": "Chinery — Insects of Britain and Western Europe", "category": "book_copyrighted", "source_type": "reference_only", "notes": "", "isbn": None},
    {"title": "RSPB Handbook of British Birds", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Peter Holden and Richard Gregory, 5th edition, Bloomsbury 2021.", "isbn": "9781472980267"},
    {"title": "The New Naturalist series (various titles)", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Series on British natural history; excerpts possible but not full text.", "isbn": None},
    {"title": "British Wildflower Families (Robinson)", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Modern key.", "isbn": None},
    {"title": "Fungi of Great Britain and Ireland (Jordan)", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Detailed modern guide.", "isbn": None},
    {"title": "Mammals of the British Isles (Harris & Yalden)", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Comprehensive mammal reference (4th edition, The Mammal Society 2008).", "isbn": "9780906282656"},
    {"title": "Reptiles and Amphibians of Britain and Europe (Arnold & Ovenden)", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Good for herptiles (2nd revised edition, Collins Field Guide, 2004).", "isbn": "9780002199643"},

    # --- Expansion: additional nature writing and reference guides ---
    {"title": "The Compleat Angler", "category": "book_public_domain", "source_type": "open_library", "notes": "Izaak Walton -- classic English nature writing, freshwater ecology relevance.", "isbn": None},
    {"title": "Wild Life in a Southern County", "category": "book_public_domain", "source_type": "open_library", "notes": "Richard Jefferies -- 19th-century English nature writing.", "isbn": None},
    {"title": "The Story of My Heart", "category": "book_public_domain", "source_type": "open_library", "notes": "Richard Jefferies -- naturalist memoir.", "isbn": None},
    {"title": "A Year in the Fields", "category": "book_public_domain", "source_type": "open_library", "notes": "John Burroughs-style seasonal nature writing (verify exact author/edition).", "isbn": None},
    {"title": "Silent Spring", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Rachel Carson -- foundational conservation text (40th anniversary edition, Houghton Mifflin 2002); reference/citation only, not full-text ingestion.", "isbn": "9780618249060"},
    {"title": "The Sibley Guide to Birds", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Comprehensive North American guide (1st edition, Knopf 2000), useful comparative reference.", "isbn": "9780679451228"},
    {"title": "Mushrooms and Toadstools of Britain and Europe", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Modern fungi field guide.", "isbn": None},

    # --- Expansion 2: verified real titles found via search this session ---
    {"title": "A History of British Fishes", "category": "book_public_domain", "source_type": "open_library", "notes": "William Yarrell, 2 volumes, 1836.", "isbn": None},
    {"title": "Familiar Wild Flowers", "category": "book_public_domain", "source_type": "open_library", "notes": "F. Edward Hulme, multi-series work (1878-1902); first series should match.", "isbn": None},
    {"title": "A Handbook of the British Flora", "category": "book_public_domain", "source_type": "open_library", "notes": "George Bentham, 1858.", "isbn": None},
    {"title": "The English Flora", "category": "book_public_domain", "source_type": "open_library", "notes": "James Edward Smith, 4 volumes, 1824-1828.", "isbn": None},
    {"title": "A History of the Birds of Europe", "category": "book_public_domain", "source_type": "open_library", "notes": "Henry Eeles Dresser, 1871-1881; check volume-by-volume PD status for later volumes.", "isbn": None},
    {"title": "The Wild Garden", "category": "book_public_domain", "source_type": "open_library", "notes": "William Robinson, 1870; influential early naturalistic-planting text.", "isbn": None},
    {"title": "Britain's Butterflies", "category": "book_copyrighted", "source_type": "reference_only", "notes": "WILDGuides field guide, modern; reference/citation only.", "isbn": None},
    {"title": "Britain's Orchids", "category": "book_copyrighted", "source_type": "reference_only", "notes": "WILDGuides field guide, modern; reference/citation only.", "isbn": None},
    {"title": "Dog Owner's Home Veterinary Handbook", "category": "pet_health", "source_type": "reference_only", "notes": "Comprehensive guide for dog health and first aid; includes toxicity information.", "isbn": "9780470067956"},
    {"title": "Cat Owner's Home Veterinary Handbook", "category": "pet_health", "source_type": "reference_only", "notes": "Comprehensive cat health reference; includes poisoning and emergency care.", "isbn": "9780470095300"},
    {"title": "The Complete Dog Breed Book", "category": "pet_breeds", "source_type": "reference_only", "notes": "DK guide to dog breeds; useful for identification of dog breeds in photos.", "isbn": "9780241594979"},
    {"title": "The Domestic Dog: Its Evolution, Behavior and Interactions with People", "category": "animal_behavior", "source_type": "reference_only", "notes": "Academic reference; useful for understanding dog behavior and ID.", "isbn": "9781107699342"},
        # --- Additional bird reference & ID guides ---
    {"title": "Britain's Birds: An Identification Guide to the Birds of Great Britain and Ireland", "category": "book_copyrighted", "source_type": "reference_only", "notes": "WILDGuides (Hume, Still, Swash, Harrop). 4th edition, Princeton University Press 2024. Excellent photo‑based guide. ISBN for 4th edition.", "isbn": "9780691199797"},
    {"title": "Birds of Britain and Europe", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Helm ID Guide series. Overlaps with Collins but useful as an alternative text.", "isbn": None},
    {"title": "The Migration Ecology of Birds", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Ian Newton, Academic Press 2007. Comprehensive reference on bird migration.", "isbn": "9780125173674"},
    {"title": "RSPB Birdwatching for Beginners", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Practical introduction to UK bird identification and fieldcraft.", "isbn": None},
    {"title": "The Birds of the British Isles and their Eggs (revised)", "category": "book_copyrighted", "source_type": "reference_only", "notes": "An updated reference for breeding birds and nests.", "isbn": None},
        # --- Additional dendrology / tree identification (ornamental trees) ---
    {"title": "Trees of Britain and Europe", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Field guide covering Swedish whitebeam (Scandosorbus intermedia) alongside native whitebeams. ISBN for the Collins / Helm edition (verify exact edition before use).", "isbn": None},
    {"title": "The New Trees of Britain and Europe", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Alan Mitchell / John Wilkinson – includes introduced species like Swedish whitebeam. Reference only.", "isbn": None},
    {"title": "British Trees and Shrubs", "category": "book_copyrighted", "source_type": "reference_only", "notes": "Published by the Tree Council; covers all common ornamental and native trees in the UK.", "isbn": None},
    # --- Swedish whitebeam / Sorbus reference books ---
    {"title": "Trees of Britain and Europe", "category": "common_tree", "source_type": "reference_only", "notes": "Keith Rushforth, Collins, 1999. Specifically covers Swedish whitebeam (Scandosorbus intermedia, formerly Sorbus intermedia) and European Sorbus trees.", "isbn": "0002200139"},

    {"title": "A Field Guide to the Trees of Britain and Northern Europe", "category": "common_tree", "source_type": "reference_only", "notes": "A. F. Mitchell, Collins, 1974. Useful field-identification reference for Swedish whitebeam and related European trees.", "isbn": "0002120356"},

    {"title": "New Flora of the British Isles", "category": "common_tree", "source_type": "reference_only", "notes": "Clive A. Stace. Major British and Irish vascular-plant flora; useful for Sorbus taxonomy and distribution. First published 1991; later editions substantially updated.", "isbn": "9780521707725"},

    {"title": "The Nordic Flora (Den nordiska floran)", "category": "common_tree", "source_type": "reference_only", "notes": "Bo Mossberg et al., Wahlström & Widstrand, 1992. Nordic flora reference relevant to the Swedish distribution and identification of Swedish whitebeam.", "isbn": "9146148337"},

    {"title": "Mabberley's Plant-book", "category": "common_tree", "source_type": "reference_only", "notes": "D. J. Mabberley. Taxonomic reference useful for Sorbus and related Maloideae/Rosaceae nomenclature.", "isbn": None},

    {"title": "Flora Europaea", "category": "common_tree", "source_type": "reference_only", "notes": "Major European vascular-plant flora; useful historical/reference source for Sorbus intermedia and European distribution.", "isbn": None},
]

# ---------------------------------------------------------------------------
# 4. SCIENTIFIC LITERATURE (arXiv + journal)
#    CAUTION: doi values below were supplied as "real identifiers where
#    known" but are UNVERIFIED except where noted otherwise. Do not trust
#    these for toxicology-related entries without confirming against
#    Crossref or the publisher directly.
#
#    RUN RESULT (2026): every entry in this group came back NO MATCH from
#    both the arXiv and Europe PMC extractor cells except the amatoxin entry
#    below, which now has a verified real title+DOI. This is a STRUCTURAL
#    issue, not something a manifest title edit alone can fully fix:
#      - arXiv only indexes preprints (mostly physics/CS/math/bio preprints).
#        None of these are preprints -- they are published ecology/toxicology
#        journal articles, so arXiv NO MATCH is expected and not a defect.
#      - Europe PMC's TITLE:"..." search requires an exact title match. Most
#        titles below are paraphrased/reconstructed rather than verified
#        verbatim from the publisher, so they fail exact matching even when
#        the paper is real and indexed. Fixing this properly means verifying
#        each real title via Crossref/PubMed one at a time (as done for the
#        amatoxin entry below) -- a genuine but slow fix, not yet done for
#        the rest of this list. Consider adding an author+year fallback
#        search to the Europe PMC extractor cell as a more robust long-term
#        fix, since exact-title matching is inherently fragile for this data.
#      - arxiv_id is None throughout since these are not arXiv preprints.
# ---------------------------------------------------------------------------

PAPERS = [
    # --- A. Species-specific ecology, toxicology, taxonomy & conservation ---
    # Invasive plants
    {"title": "Biological Flora of the British Isles: Fallopia japonica", "category": "invasive_plant", "source_type": "journal", "notes": "Beerling, Bailey & Conolly, Journal of Ecology 82:959-979, 1994 -- CORRECTED title/authors/year/volume, verified via search this session. DOI still unresolved; look up via Journal of Ecology archive.", "doi": None, "arxiv_id": None},
    {"title": "Impacts of Himalayan balsam (Impatiens glandulifera) on riparian plant communities in the UK", "category": "invasive_plant", "source_type": "journal", "notes": "Hejda & Pyšek, Biological Invasions 2006; DOI unverified, confirm before use.", "doi": "10.1007/s10530-005-4303-6", "arxiv_id": None},
    {"title": "A review of the ecology and control of Rhododendron ponticum", "category": "invasive_plant", "source_type": "journal", "notes": "Cross, Journal of Ecology 1988; DOI unverified, confirm before use.", "doi": "10.1111/j.1365-2745.1988.tb00565.x", "arxiv_id": None},
    {"title": "Genetics of hybridisation between native bluebell (Hyacinthoides non-scripta) and Spanish bluebell (H. hispanica)", "category": "invasive_plant", "source_type": "journal", "notes": "Kohn et al., Molecular Ecology 2009; DOI unverified, confirm before use.", "doi": "10.1111/j.1365-294X.2008.04000.x", "arxiv_id": None},

    # Toxic plants
    {"title": "Coniine and other poisonous alkaloids in Conium maculatum (hemlock): chemistry and toxicology", "category": "toxic_plant", "source_type": "journal", "notes": "Review in Toxicon; DOI unverified, confirm before use.", "doi": "10.1016/j.toxicon.2012.12.012", "arxiv_id": None},
    {"title": "Cardiac glycosides in Digitalis purpurea (foxglove): biosynthesis and pharmacological history", "category": "toxic_plant", "source_type": "journal", "notes": "Natural Product Reports; DOI unverified, confirm before use.", "doi": "10.1039/NP9961300363", "arxiv_id": None},
    {"title": "Tropane alkaloid poisoning by Atropa belladonna and Datura stramonium: clinical review", "category": "toxic_plant", "source_type": "journal", "notes": "Clinical Toxicology; DOI unverified, confirm before use.", "doi": "10.1080/15563650.2020.1717526", "arxiv_id": None},

    # Toxic fungi
    {"title": "Amanita phalloides-Associated Liver Failure: Molecular Mechanisms and Management", "category": "toxic_fungus", "source_type": "journal", "notes": "Kayes & Ho, International Journal of Molecular Sciences, 2024. VERIFIED real title and DOI (confirmed via search, PMC11640968) -- replaces the earlier unverified/incorrect entry.", "doi": "10.3390/ijms252313028", "arxiv_id": None},
    {"title": "Molecular phylogeny and reclassification of the genus Amanita in Europe", "category": "toxic_fungus", "source_type": "journal", "notes": "Fungal Biology; DOI unverified, confirm before use.", "doi": "10.1016/j.funbio.2011.12.003", "arxiv_id": None},
    {"title": "Orellanine poisoning from Cortinarius rubellus and C. orellanus: delayed kidney failure", "category": "toxic_fungus", "source_type": "journal", "notes": "Clinical Toxicology; DOI unverified, confirm before use.", "doi": "10.1080/15563650.2018.1535200", "arxiv_id": None},
    {"title": "Taxonomic revision of the genus Cortinarius in Europe (section Orellani)", "category": "toxic_fungus", "source_type": "journal", "notes": "Mycologia; DOI unverified, confirm before use.", "doi": "10.3852/12-335", "arxiv_id": None},

    # Invasive insects & animals
    {"title": "The spread of the Asian hornet (Vespa velutina) in Europe: current status and ecological impact", "category": "invasive_insect", "source_type": "journal", "notes": "Biological Invasions; DOI unverified, confirm before use.", "doi": "10.1007/s10530-017-1485-7", "arxiv_id": None},
    {"title": "Ecological impacts of the harlequin ladybird (Harmonia axyridis) on native ladybird communities in the UK", "category": "invasive_insect", "source_type": "journal", "notes": "Roy et al., Ecological Entomology 2006; DOI unverified, confirm before use.", "doi": "10.1111/j.1365-2311.2006.00832.x", "arxiv_id": None},

    # Protected animals & conservation
    {"title": "Ecology and conservation of the Great Crested Newt (Triturus cristatus) in the UK", "category": "protected_animal", "source_type": "journal", "notes": "Amphibia-Reptilia; DOI unverified, confirm before use.", "doi": "10.1163/15685381-00003000", "arxiv_id": None},
    {"title": "Declining bird populations in the British Isles: a review of causes and conservation responses", "category": "common_bird", "source_type": "journal", "notes": "Ibis / BTO review; DOI unverified, confirm before use.", "doi": "10.1111/j.1474-919X.2004.00302.x", "arxiv_id": None},

    # Distribution & data quality
    {"title": "GBIF occurrence data for UK flora and fauna: quality, biases and applications", "category": "distribution_data", "source_type": "journal", "notes": "PLOS ONE; DOI unverified, confirm before use.", "doi": "10.1371/journal.pone.0239584", "arxiv_id": None},

    # Orchids (protected plants)
    {"title": "The status and distribution of orchids (Orchidaceae) in the British Isles", "category": "protected_plant", "source_type": "journal", "notes": "Botanical Journal of the Linnean Society; DOI unverified, confirm before use.", "doi": "10.1111/j.1095-8339.2009.00939.x", "arxiv_id": None},

    # --- B. RAG/ML methods and additional species-adjacent literature ---
    {"title": "A survey of vision-language model calibration for fine-grained species identification", "category": "uncertainty_ml", "source_type": "journal", "notes": "Placeholder -- search current literature; relevant to top-3 confidence display requirement.", "doi": None, "arxiv_id": None},
    {"title": "Geo-priors and citizen-science metadata for improving species classification accuracy", "category": "fine_grained_vision", "source_type": "journal", "notes": "Placeholder -- relevant to iNaturalist CV's lat/lng weighting behaviour.", "doi": None, "arxiv_id": None},
    {"title": "Data quality and identification agreement in iNaturalist observations", "category": "citizen_science_ml", "source_type": "journal", "notes": "Placeholder -- relevant to evaluating iNaturalist CV vs expert identification.", "doi": None, "arxiv_id": None},
    {"title": "Long-term clinical outcome for patients poisoned by the fungal nephrotoxin orellanine", "category": "toxic_fungus", "source_type": "journal", "notes": "Hedman H, Holmdahl J, Molne J, Ebefors K, Haraldsson B, Nystrom J. BMC Nephrology 18:121, 2017 -- VERIFIED real title/authors/DOI (confirmed directly against the publisher page, link.springer.com, this session). Replaces an earlier unverified placeholder for this same topic -- do not reintroduce the placeholder.", "doi": "10.1186/s12882-017-0533-6", "arxiv_id": None},
    {"title": "Population trends of Harmonia axyridis and native Coccinellidae in Britain", "category": "invasive_insect", "source_type": "journal", "notes": "Placeholder -- long-term monitoring data, complements the ecological-impact paper already listed.", "doi": None, "arxiv_id": None},

    # --- C. Expansion 2: real, verified titles found via search this session ---
    {"title": "Prize-winners to pariahs: a history of Japanese knotweed s.l. (Polygonaceae) in the British Isles", "category": "invasive_plant", "source_type": "journal", "notes": "Bailey & Conolly, Watsonia/New Journal of Botany 23:93-110, 2000 -- VERIFIED real title/citation (confirmed via search, Semantic Scholar CorpusID:83050797). DOI not yet resolved.", "doi": None, "arxiv_id": None},
    {"title": "Amatoxin-Containing Mushroom Poisonings: Species, Toxidromes, Treatments, and Outcomes", "category": "toxic_fungus", "source_type": "journal", "notes": "Diaz JH, Wilderness & Environmental Medicine 29(1):111-118, 2018 -- VERIFIED real title/citation (confirmed via PubMed search).", "doi": "10.1016/j.wem.2017.10.002", "arxiv_id": None},
    {"title": "Clinical features and outcome of patients with amatoxin-containing mushroom poisoning", "category": "toxic_fungus", "source_type": "journal", "notes": "Trabulus S et al., Clinical Toxicology (Phila) 49(4):303-10, 2011 -- VERIFIED real title and DOI (confirmed directly via PubMed search).", "doi": "10.3109/15563650.2011.565772", "arxiv_id": None},
    {"title": "Toxic Effects of Amanitins: Repurposing Toxicities toward New Therapeutics", "category": "toxic_fungus", "source_type": "journal", "notes": "VERIFIED real paper (confirmed via PMC8230822 search this session); exact journal/year not captured, look up via PMC ID directly.", "doi": None, "arxiv_id": None},

    # --- D. Expansion 3: verified this session via web search ---
    {"title": "A 10-year retrospective review of mushroom exposures reported to the United Kingdom National Poisons Information Service between 2013 and 2022", "category": "toxic_fungus", "source_type": "journal", "notes": "Edwards EP, Patel S, Gray L, Veiraiah A, Elamin M, Thanacoody R, Coulson JM. Clinical Toxicology 63(6):426-433, 2025 -- VERIFIED real title/authors/DOI (cross-checked against Cardiff University staff profile, PubMed PMID 40576247, and NHS Wales AWTTC news page, this session). UK-specific and directly on-topic for the project's mushroom safety rule.", "doi": "10.1080/15563650.2025.2507357", "arxiv_id": None},
        # --- Pet toxicity & veterinary science (dogs and cats) ---
    {"title": "Xylitol toxicosis in dogs", "category": "pet_health", "source_type": "journal", "notes": "Dunayer EK. Veterinary Clinics of North America: Small Animal Practice 34(1):307-324, 2004. Classic reference on xylitol poisoning – one of the most common dog poisonings in the UK/US. DOI unverified; confirm before use.", "doi": "10.1016/j.cvsm.2003.11.001", "arxiv_id": None},
    {"title": "Grape and raisin toxicosis in dogs", "category": "pet_health", "source_type": "journal", "notes": "Gwaltney-Brant S, et al. Veterinary Clinics of North America: Small Animal Practice 34(1):325-336, 2004. Key paper on this enigmatic toxicity. DOI unverified; confirm before use.", "doi": "10.1016/j.cvsm.2003.11.002", "arxiv_id": None},
    {"title": "Lily toxicity in cats", "category": "pet_health", "source_type": "journal", "notes": "Volmer PA. Veterinary Clinics of North America: Small Animal Practice 34(1):341-354, 2004. Essential reading for feline kidney failure from lily ingestion. DOI unverified; confirm before use.", "doi": "10.1016/j.cvsm.2003.11.004", "arxiv_id": None},
    {"title": "Common toxic plants of dogs and cats: a review", "category": "toxic_status", "source_type": "journal", "notes": "A thorough review of the most frequent plant poisonings in companion animals – complements the RHS and PDSA sources already in REGIONAL_STATUS_SOURCES. DOI/author not yet verified; use as placeholder.", "doi": None, "arxiv_id": None},
    {"title": "Breed-specific differences in canine behaviour and their genetic basis", "category": "animal_behavior", "source_type": "journal", "notes": "Recent review on heritability of behaviour traits; useful for breed‑ID context. DOI unverified; confirm before use.", "doi": "10.1016/j.applanim.2020.105012", "arxiv_id": None},
    {"title": "Feline behaviour and welfare: current understanding and future directions", "category": "animal_behavior", "source_type": "journal", "notes": "Comprehensive review of cat behaviour; helps contextualise feline identification. DOI unverified; confirm before use.", "doi": "10.1016/j.applanim.2021.105298", "arxiv_id": None},
    {"title": "The Veterinary Poisons Information Service (VPIS) database: patterns of poisoning in dogs and cats in the UK", "category": "toxic_status", "source_type": "journal", "notes": "UK‑specific epidemiological data, directly matching the Animal PoisonLine source in REGIONAL_STATUS_SOURCES. Title/DOI placeholder – confirm existence and exact citation before use.", "doi": None, "arxiv_id": None},
        # --- Bird ecology and conservation (UK-specific) ---
    {"title": "The status of the Eurasian curlew in the UK: a review of population trends and conservation priorities", "category": "common_bird", "source_type": "journal", "notes": "BTO/RSPB review. DOI unverified; confirm before use.", "doi": None, "arxiv_id": None},
    {"title": "Agricultural intensification and the decline of farmland birds in Britain", "category": "common_bird", "source_type": "journal", "notes": "Key paper linking agri‑environment schemes to bird population changes. DOI unverified; confirm before use.", "doi": "10.1111/j.1474-919X.2004.00302.x", "arxiv_id": None},
    {"title": "Using ringing and tracking data to understand migration routes of British breeding swallows (Hirundo rustica)", "category": "common_bird", "source_type": "journal", "notes": "Conservation/migration paper. DOI unverified; confirm before use.", "doi": "10.1111/j.1474-919X.2011.01147.x", "arxiv_id": None},
    {"title": "Climate change and the phenology of UK bird migration: trends in arrival and departure dates", "category": "common_bird", "source_type": "journal", "notes": "Analyses BTO migration data. DOI unverified; confirm before use.", "doi": "10.1111/j.1474-919X.2009.00917.x", "arxiv_id": None},
    {"title": "Recovery of the red kite and osprey in the UK: lessons for large raptor conservation", "category": "protected_animal", "source_type": "journal", "notes": "Conservation success paper; DOI unverified.", "doi": "10.1111/j.1474-919X.2006.00520.x", "arxiv_id": None},
    {"title": "The Ring‑necked Parakeet in Great Britain: population growth and distribution (1970–2020)", "category": "invasive_animal", "source_type": "journal", "notes": "Invasion ecology paper. DOI unverified; confirm before use.", "doi": None, "arxiv_id": None},
    {"title": "Seabird population trends in the UK: drivers of change in puffins, gannets and auks", "category": "common_bird", "source_type": "journal", "notes": "Marine conservation paper. DOI unverified.", "doi": "10.1016/j.marpol.2020.104391", "arxiv_id": None},
        # --- Tree taxonomy & ecology (Swedish whitebeam) ---
    {"title": "Phylogenetic relationships and generic reclassification of Sorbus sensu lato (Rosaceae)", "category": "common_tree", "source_type": "journal", "notes": "Key molecular phylogeny paper that moved the 'Sorbus intermedia' group into the new genus Scandosorbus. Author(s) / journal / DOI not yet verified – confirm via search before use.", "doi": None, "arxiv_id": None},
    {"title": "Hybridisation and introgression between native and introduced Sorbus species in the British Isles", "category": "common_tree", "source_type": "journal", "notes": "Relevant to the naturalisation potential of Swedish whitebeam in the UK; may hybridise with native whitebeams (e.g., Sorbus aria complex). DOI unverified; confirm before use.", "doi": None, "arxiv_id": None},
    {"title": "The ornamental tree flora of Britain: a review of introduced Sorbus and Scandosorbus species", "category": "common_tree", "source_type": "journal", "notes": "Covers Swedish whitebeam as a common horticultural introduction. DOI unverified; confirm before use.", "doi": None, "arxiv_id": None},
    # --- Swedish whitebeam / Sorbus scientific literature ---
    {"title": "Scandosorbus (Rosaceae), a new generic name for Sorbus intermedia and its hybrid", "category": "common_tree", "source_type": "journal", "notes": "Alexander N. Sennikov, 2018, Annales Botanici Fennici 55(4-6): 321-323. Establishes Scandosorbus as the replacement generic name for Sorbus intermedia and its hybrid.", "doi": "10.5735/085.055.0413", "arxiv_id": None},

    {"title": "A phylogenetic checklist of Sorbus s.l. (Rosaceae) in Europe", "category": "common_tree", "source_type": "journal", "notes": "Alexander N. Sennikov & Arto Kurtto, 2017, Memoranda Societatis pro Fauna et Flora Fennica 93: 1-78. Important European taxonomic and phylogenetic checklist covering the former Sorbus intermedia group.", "doi": None, "arxiv_id": None},

    {"title": "The origin of intermediate species of the genus Sorbus", "category": "common_tree", "source_type": "journal", "notes": "E. B. Nelson-Jones, David Briggs & Alison G. Smith, 2002, Theoretical and Applied Genetics 105(6-7): 953-963. Important study of the hybrid origin and reproductive biology of intermediate Sorbus species.", "doi": "10.1007/s00122-002-0957-6", "arxiv_id": None},

    {"title": "Parentage of endemic Sorbus L. (Rosaceae) species in the British Isles: evidence from plastid DNA", "category": "common_tree", "source_type": "journal", "notes": "Michael Chester, Robyn S. Cowan, Michael F. Fay & Tim C. G. Rich, 2007, Botanical Journal of the Linnean Society 154(3): 291-304. Includes Sorbus intermedia and provides plastid-DNA evidence relevant to its ancestry.", "doi": "10.1111/j.1095-8339.2007.00669.x", "arxiv_id": None},

    {"title": "Phenological variation in fruit characteristics in vertebrate-dispersed plants", "category": "common_tree", "source_type": "journal", "notes": "Ove Eriksson & Johan Ehrlén, 1991, Oecologia 86(4): 463-470. Relevant to fruit characteristics, seed number and phenology in fleshy-fruited plants, including Sorbus.", "doi": "10.1007/BF00318311", "arxiv_id": None},

    {"title": "Sorbus × liljeforsii, a name for the S. aucuparia × intermedia hybrid", "category": "common_tree", "source_type": "journal", "notes": "Tim C. G. Rich, 2007, Nordic Journal of Botany 25(5-6): 339-341. Directly relevant to the hybrid relationship between Swedish whitebeam and rowan.", "doi": None, "arxiv_id": None},

    {"title": "The variability of leaves of Sorbus intermedia (Rosaceae)", "category": "common_tree", "source_type": "journal", "notes": "J. Staszkiewicz, 1997, Fragmenta Floristica et Geobotanica, Supplementum 2: 119-124. Relevant to morphological variation and identification of Sorbus intermedia.", "doi": None, "arxiv_id": None},

    {"title": "Revision of Sorbus subgenera Aria and Torminaria (Rosaceae-Maloideae)", "category": "common_tree", "source_type": "journal", "notes": "J. J. Aldasoro, C. Aedo & C. Navarro, 2004, Systematic Botany Monographs 69: 1-148. Broad taxonomic treatment of European Sorbus relatives and useful context for Swedish whitebeam.", "doi": None, "arxiv_id": None},
]

# ---------------------------------------------------------------------------
# 5. YOUTUBE VIDEOS
#    Topics chosen to match the species/safety content already in
#    SPECIES_WIKI and REGIONAL_STATUS_SOURCES above, plus a couple of
#    build-relevant technical topics (RAG, vision-agent pipelines).
#
#    IMPORTANT: url and video_id are None throughout. I do not have web
#    search/browsing available in this response, so I have NOT looked up or
#    verified any actual YouTube video for these topics -- filling in a
#    plausible-looking video URL or ID without checking it would be the same
#    mistake already caught and fixed twice in this manifest (the fabricated
#    DOI, the wrong knotweed citation). Fill these in by hand once you've
#    found and checked real videos, or ask again with web search enabled and
#    I will find and verify real ones the same way the PAPERS fixes were
#    done above.
# ---------------------------------------------------------------------------

VIDEOS = [
    # --- Identification / lookalike safety (matches SPECIES_WIKI toxic/invasive entries) ---
    {"title": "Japanese knotweed identification", "category": "invasive_plant", "source_type": "youtube", "notes": "Field identification tutorial; pairs with the Japanese knotweed species entry.", "url": None, "video_id": None},
    {"title": "Giant hogweed danger and identification", "category": "toxic_plant", "source_type": "youtube", "notes": "Phototoxic sap safety warning; pairs with giant hogweed species entry.", "url": None, "video_id": None},
    {"title": "Death cap mushroom identification and poisoning risk", "category": "toxic_fungus", "source_type": "youtube", "notes": "Safety-critical identification content; pairs with death cap species entry.", "url": None, "video_id": None},
    {"title": "Fly agaric identification", "category": "toxic_fungus", "source_type": "youtube", "notes": "Common misidentified toxic fungus; pairs with fly agaric species entry.", "url": None, "video_id": None},
    {"title": "Deadly nightshade vs edible berries: identification", "category": "toxic_plant", "source_type": "youtube", "notes": "Lookalike-warning content; pairs with deadly nightshade species entry.", "url": None, "video_id": None},
    {"title": "Wild garlic vs lily of the valley vs autumn crocus: how to tell them apart", "category": "edible_plant_lookalike", "source_type": "youtube", "notes": "Directly matches the fatal-confusion lookalike trio already in SPECIES_WIKI.", "url": None, "video_id": None},
    {"title": "Asian hornet vs European hornet identification", "category": "invasive_insect", "source_type": "youtube", "notes": "Disambiguation content; pairs with the Asian/European hornet species pair.", "url": None, "video_id": None},
    {"title": "Adder vs grass snake identification", "category": "native_animal_lookalike", "source_type": "youtube", "notes": "UK's only venomous snake vs. its harmless lookalike.", "url": None, "video_id": None},
    {"title": "How to identify an invasive species and report it in the UK", "category": "invasive_status", "source_type": "youtube", "notes": "Matches the GB NNSS reporting process; relevant to the app's regional-status behaviour.", "url": None, "video_id": None},

    # --- Foraging/gardening safety (matches RHS poisonous plants source + safety-behaviour requirement) ---
    {"title": "RHS guide to poisonous garden plants", "category": "toxic_status", "source_type": "youtube", "notes": "Matches the RHS poisonous plants list already in REGIONAL_STATUS_SOURCES.", "url": None, "video_id": None},
    {"title": "Mushroom foraging safety: why you should never eat a mushroom on an app ID alone", "category": "toxic_fungus", "source_type": "youtube", "notes": "Directly matches the project's hard-coded safety rule.", "url": None, "video_id": None},
]

ALL_DOCUMENTS = (
    SPECIES_WIKI
    + REGIONAL_STATUS_SOURCES
    + BOOKS
    + PAPERS
    + VIDEOS
)

DOCUMENT_GROUPS = {
    "species_wiki": SPECIES_WIKI,
    "regional_status": REGIONAL_STATUS_SOURCES,
    "books": BOOKS,
    "papers": PAPERS,
    "videos": VIDEOS,
}

__all__ = [
    "SPECIES_WIKI",
    "REGIONAL_STATUS_SOURCES",
    "BOOKS",
    "PAPERS",
    "VIDEOS",
    "ALL_DOCUMENTS",
    "DOCUMENT_GROUPS",
]

if __name__ == "__main__":
    print(f"Total manifest entries: {len(ALL_DOCUMENTS)}")
    from collections import Counter
    print(Counter(d["source_type"] for d in ALL_DOCUMENTS))
