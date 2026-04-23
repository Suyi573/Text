import json
from pathlib import Path

def load_chunk_index(corpus_path):
    import json

    chunk_index = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            chunk_id = item["chunk_id"]
            metadata = item.get("metadata", {})

            chunk_index[chunk_id] = {
                "doc_id": metadata.get("doc_id"),
                "source": metadata.get("source"),
                "title": metadata.get("title"),
                "url": metadata.get("url"),
            }

    return chunk_index

corpus_path = Path("data/processed/chunks_v3.jsonl")
chunk_index = load_chunk_index(corpus_path)


def generate_benchmark():
    """
    Benchmark generator:
    - Flat structure (no metadata nesting)
    - Adds answerable flag
    - Uses relevant_chunk_ids (primary)
    - Keeps relevant_doc_ids (secondary)
    - Uses stable query_id format
    """

    output_path = Path("data/benchmark/benchmark.json")
    input_path = Path("data/benchmark/input_payload.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_data = [
        # =========================
        # EASY
        # =========================
        {
            "query_id": "q_000",
            "query": "The name 'bouillabaisse' is derived from which two words?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_10"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "bolhir and abaisser",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_001",
            "query": "Which two ingredients are traditionally not used in authentic Italian carbonara?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_199"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Cream and garlic",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_002",
            "query": "What is chermoula?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_83"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "A North African marinade made from herbs, oil, spices, and preserved lemon",
            "difficulty": "easy",
            "type": "definition",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_003",
            "query": "What is passata in Italian cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_162"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "A smooth tomato purée made from strained and peeled tomatoes",
            "difficulty": "easy",
            "type": "definition",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_004",
            "query": "What are the three core components of the Mediterranean dietary trinity?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_613", "chunk_627", "chunk_381", "chunk_384"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Olive oil, wheat, and grapes",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "mixed"
        },
        {
            "query_id": "q_005",
            "query": "In Trapani (Sicily), which North African-influenced dish is commonly served with fish?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_535"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Couscous",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_006",
            "query": "What dish is traditionally eaten on Thursdays in the Lazio region of Italy?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_508"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Gnocchi",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_007",
            "query": "What health benefit is commonly associated with drinking karkadeh in Egypt?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_364"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It is believed to help lower blood pressure",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "blog80cuisines"
        },
        {
            "query_id": "q_008",
            "query": "How are French and Italian cuisines contrasted?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_2"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Italian cuisine is described as simple and focused on fresh ingredients, whereas French cuisine is characterized by complexity and refined techniques",
            "difficulty": "easy",
            "type": "comparison",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_009",
            "query": "Which plant is known as the bread tree in Mediterranean mountain regions?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_644"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "The chestnut tree",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_010",
            "query": "What type of flour is commonly used to fry fish in the Black Sea region?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_1105"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Thick corn flour",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikibooks"
        },
        {
            "query_id": "q_011",
            "query": "What are the typical fillings of su böreği in Turkish cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_1113"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Boiled yufka layers, cheese and parsley",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikibooks"
        },
        {
            "query_id": "q_012",
            "query": "Why did the Romans forbid growing vines in Provence in 120 BC?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_961"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "To protect the profitable trade of exporting Italian wines",
            "difficulty": "easy",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_013",
            "query": "Why can olive oil replace saturated fats?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_688"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Because it is rich in oleic acid, which helps maintain normal low-density lipoprotein cholesterol (LDL-C) levels",
            "difficulty": "easy",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_014",
            "query": "What causes a soufflé to rise during baking?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_17"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It rises due to the expansion of air in whipped egg whites when heated",
            "difficulty": "easy",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_015",
            "query": "Which herb is used extensively in Sicilian cooking compared to the rest of Italy?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_535"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Mint",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_016",
            "query": "What tree is used to define the Mediterranean region?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_620"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "The olive tree",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_017",
            "query": "What type of dish is dolma?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_150"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Stuffed vegetables",
            "difficulty": "easy",
            "type": "definition",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_018",
            "query": "What are some common ingredients in Greek desserts?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_382"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Nuts, honey, fruits, sesame, and filo pastries",
            "difficulty": "easy",
            "type": "factoid",
            "source_category": "wikipedia"
        },

        # =========================
        # MEDIUM
        # =========================
        {
            "query_id": "q_019",
            "query": "What happens to Asturian cider when it is poured using the traditional escanciada technique?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_1012"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "When poured from a height, the cider breaks as it hits the glass, becoming aerated and bubbly",
            "difficulty": "medium",
            "type": "process",
            "source_category": "wikibooks"
        },
        {
            "query_id": "q_020",
            "query": "How does the pizza Quattro Stagioni represent the four seasons through its toppings?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_175"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Different ingredients are used to symbolize each season, such as tomato and basil, mushrooms and truffles, ham and olives, and artichokes",
            "difficulty": "medium",
            "type": "reasoning",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_021",
            "query": "How does the Healthy Mediterranean-Style Eating Pattern differ from the Healthy U.S.-Style Eating Pattern?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_690"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It recommends more fruits and seafood, and less dairy",
            "difficulty": "medium",
            "type": "comparison",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_022",
            "query": "What is the key difference between dolma and sarma in Turkish cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_1122", "chunk_1126"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Dolma refers to ingredients stuffed inside vegetables, while sarma involves wrapping fillings",
            "difficulty": "medium",
            "type": "comparison",
            "source_category": "wikibooks"
        },
        {
            "query_id": "q_023",
            "query": "What three conditions indicate that a paella is done?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_1171", "chunk_1172"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "The rice should be slightly firm to the bite, the paella should be moist but not soupy, and there should be some toasted rice at the bottom of the pan",
            "difficulty": "medium",
            "type": "factoid",
            "source_category": "wikibooks"
        },
        {
            "query_id": "q_024",
            "query": "Why are indentations made in focaccia dough during preparation in Ligurian cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_255"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "To prevent large air pockets from forming and to allow oil and seasonings to collect, helping keep the bread moist",
            "difficulty": "medium",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_025",
            "query": "What do early Egyptian cheese remains tell us about how cheese was made?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_333"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "They suggest that early cheese was made using acid or a combination of acid and heat",
            "difficulty": "medium",
            "type": "inference",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_026",
            "query": "How does arroz negro reflect Mediterranean seafood cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_1159", "chunk_1161"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It combines rice with seafood like squid and uses ingredients such as olive oil and seafood broth",
            "difficulty": "medium",
            "type": "reasoning",
            "source_category": "wikibooks"
        },
        {
            "query_id": "q_027",
            "query": "What characterizes the typical ingredients used in Sicilian cooking?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_535"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "A combination of fresh vegetables and a wide variety of seafood",
            "difficulty": "medium",
            "type": "summary",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_028",
            "query": "What is the philosophy behind cucina povera in Tuscan cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_543"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It focuses on using simple, seasonal, and inexpensive ingredients rather than complex cooking techniques",
            "difficulty": "medium",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_029",
            "query": "Why is Mediterranean cuisine considered culturally unified?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_622"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Because it reflects shared practices and ingredients that transcend differences in language, religion, and society",
            "difficulty": "medium",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_030",
            "query": "How did Greek cuisine influence other regions?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_383"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It spread to ancient Rome and later influenced European cuisines",
            "difficulty": "medium",
            "type": "history",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_031",
            "query": "What factors contributed to the development of Italian cuisine over time?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_413"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Trade, conquests, political changes, and the discovery of the New World all influenced its evolution",
            "difficulty": "medium",
            "type": "history",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_032",
            "query": "What is the key difference between the Mediterranean diet and Mediterranean cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_616"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "The diet focuses on health, while cuisine refers to cooking practices and food preparation",
            "difficulty": "medium",
            "type": "comparison",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_033",
            "query": "Why is Turkish cuisine described as especially diverse in flavour and influence?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_42"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Because it sits at the crossroads of many culinary traditions, including Middle Eastern, Greek, Balkan, Caucasian, and Asian influences, and has a long history of trade and culinary exchange",
            "difficulty": "medium",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_034",
            "query": "Why are pigs and dogs both used in truffle hunting, and what is the trade-off between them?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_197", "chunk_198"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Pigs naturally detect truffles because of a compound similar to a boar pheromone, but they may eat them when found, while dogs must be trained but are less likely to damage the harvest",
            "difficulty": "medium",
            "type": "comparison",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_035",
            "query": "Why is authentic paella not supposed to be stirred once the rice has been added?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_239", "chunk_240"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Because leaving it unstirred helps form the socarrat, the flavourful crust on the bottom of the rice, and stirring would ruin that hallmark feature",
            "difficulty": "medium",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_036",
            "query": "How does ancient Greek cuisine reflect a classic Mediterranean eating pattern?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_384"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It was based on wheat, olive oil, and wine, with meat eaten rarely and fish consumed more often, which matches a frugal Mediterranean pattern centered on staple crops and olive oil",
            "difficulty": "medium",
            "type": "reasoning",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_037",
            "query": "Why are fish dishes more common than beef dishes in Greek cuisine?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_390"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Because the Greek climate and terrain favour the breeding of goats and sheep rather than cattle, making beef less common, while fish is widely eaten in coastal areas and on the islands",
            "difficulty": "medium",
            "type": "why",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_038",
            "query": "Why is the tagine pot's design particularly well-suited to the North African environment and food processing needs?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_76", "chunk_77", "chunk_78"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "The tagine pot has a shallow bottom and a conical lid, which effectively retains steam and reduces moisture loss, important in arid environments. The small groove on the lid allows for the addition of cold water, further helping to seal in steam",
            "difficulty": "medium",
            "type": "reasoning",
            "source_category": "wikipedia"
        },

        # =========================
        # HARD
        # =========================
        {
            "query_id": "q_039",
            "query": "How was the reputation of Provence wines viewed historically compared to more recent times?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_962", "chunk_963", "chunk_964"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "They were once criticized as poor quality and ordinary, but later improved due to better cultivation and technology",
            "difficulty": "hard",
            "type": "comparison",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_040",
            "query": "How is Mediterranean cuisine both unified and diverse?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_625", "chunk_624", "chunk_627"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "It is unified by shared ingredients and historical roots, but highly diverse in practice, as regional cuisines differ significantly across the Mediterranean",
            "difficulty": "hard",
            "type": "multi_hop",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_041",
            "query": "What changes in staple ingredients and drinks illustrate the transition of Egyptian cuisine during the Greco-Roman period?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_304", "chunk_305"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Traditional grains were replaced by wheat, olive oil replaced radish oil, and wine became more popular alongside beer, reflecting dietary transformation under foreign influence",
            "difficulty": "hard",
            "type": "multi_hop",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_042",
            "query": "Why did bacalhau become such an important ingredient in Portuguese cuisine despite Portugal’s strong access to fresh fish?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_112", "chunk_113", "chunk_114"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Because salted cod was practical for long sea voyages without refrigeration during the era when Portugal was heavily involved in overseas exploration and colonisation, and over time it became an iconic national ingredient",
            "difficulty": "hard",
            "type": "multi_hop",
            "source_category": "blog80cuisines"
        },
        {
            "query_id": "q_043",
            "query": "What are the proven health benefits of olive oil?",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_688", "chunk_689"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "The polyphenols in olive oil can prevent lipid oxidation, and its oleic acid can maintain normal low-density lipoprotein cholesterol levels by replacing saturated fats, thus reducing all-cause mortality and stroke risk",
            "difficulty": "hard",
            "type": "summary",
            "source_category": "wikipedia"
        },
        {
            "query_id": "q_044",
            "query": "Summarize the potential effects of the Mediterranean diet on brain health and mental state, and the limitations of the research.",
            "answerable": True,
            "relevant_chunk_ids": ["chunk_703", "chunk_704", "chunk_705"],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Studies show that high adherence to this diet is associated with better cognitive performance, a lower risk of Alzheimer's disease, and slower cognitive decline. Observational studies have also linked it to a reduced risk of depression. However, the evidence is weak and cannot prove a causal relationship",
            "difficulty": "hard",
            "type": "summary",
            "source_category": "wikipedia"
        },

        # =========================
        # NEGATIVE
        # =========================
        {
            "query_id": "q_045",
            "query": "Which Mediterranean country has the highest per capita olive oil consumption?",
            "answerable": False,
            "relevant_chunk_ids": [],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Not mentioned in the provided corpus",
            "difficulty": "negative",
            "type": "unanswerable",
            "source_category": "unknown"
        },
        {
            "query_id": "q_046",
            "query": "How many grams of olive oil are recommended daily in the Mediterranean diet?",
            "answerable": False,
            "relevant_chunk_ids": [],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Not mentioned in the provided corpus",
            "difficulty": "negative",
            "type": "unanswerable",
            "source_category": "unknown"
        },
        {
            "query_id": "q_047",
            "query": "Which Mediterranean fish is considered the most nutritious?",
            "answerable": False,
            "relevant_chunk_ids": [],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Not mentioned in the provided corpus",
            "difficulty": "negative",
            "type": "unanswerable",
            "source_category": "unknown"
        },
        {
            "query_id": "q_048",
            "query": "Which country's desserts are most preferred across the Mediterranean region?",
            "answerable": False,
            "relevant_chunk_ids": [],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Not mentioned in the provided corpus",
            "difficulty": "negative",
            "type": "unanswerable",
            "source_category": "unknown"
        },
        {
            "query_id": "q_049",
            "query": "Who originally invented the Greek salad?",
            "answerable": False,
            "relevant_chunk_ids": [],
            "relevant_doc_ids": [],
            "gold_standard_answer": "Not mentioned in the provided corpus",
            "difficulty": "negative",
            "type": "unanswerable",
            "source_category": "unknown"
        },
    ]

    for item in benchmark_data:
        doc_ids = []
        sources = []

        for chunk_id in item["relevant_chunk_ids"]:
            if chunk_id not in chunk_index:
                print(f"Warning: {chunk_id} not found in corpus index")
                continue

            meta = chunk_index[chunk_id]

            doc_id = meta.get("doc_id")
            source = meta.get("source")

            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)

            if source and source not in sources:
                sources.append(source)

        item["relevant_doc_ids"] = doc_ids

        if len(sources) == 1:
            item["source_category"] = sources[0]
        elif len(sources) > 1:
            item["source_category"] = "mixed"
        else:
            item["source_category"] = "unknown"

    # Save benchmark (with answers)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "mediterranean_rag_benchmark_v1",
            "cuisine_scope": "Mediterranean",
            "version": "1.0",
            "items": benchmark_data
        }, f, indent=2, ensure_ascii=False)

    # Build input payload (for inference)
    input_payload = {
        "queries": [
            {
                "query_id": item["query_id"],
                "query": item["query"]
            }
            for item in benchmark_data
        ]
    }

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_payload, f, indent=2, ensure_ascii=False)

    print(f"Benchmark saved to {output_path}")
    print(f"Input payload saved to {input_path}")
    print(f"Total benchmark items: {len(benchmark_data)}")


if __name__ == "__main__":
    generate_benchmark()