import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


def init_ontology():
    print(" Inizializzazione Ontologia Knowledge Graph - Guida Turistica Giappone")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # ==========================================
    # 1. CONSTRAINT DI UNICITÀ (SOLO SUI NOMI)
    # ==========================================
    constraints = [
        "CREATE CONSTRAINT region_name IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE",
        "CREATE CONSTRAINT prefecture_name IF NOT EXISTS FOR (p:Prefecture) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE", 
        "CREATE CONSTRAINT spot_name IF NOT EXISTS FOR (s:Spot) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT festival_name IF NOT EXISTS FOR (f:Festival) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT localfood_name IF NOT EXISTS FOR (l:LocalFood) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT craft_name IF NOT EXISTS FOR (c:Craft) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT blogpost_title IF NOT EXISTS FOR (b:BlogPost) REQUIRE b.title IS UNIQUE",
        "CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE",
    ]

    with driver.session() as session:
        for query in constraints:
            session.run(query)
    print("Constraints di unicità applicati con successo.")

    # ==========================================
    # 2. SEED: REGIONI (8)
    # ==========================================
    regions = [
        "Hokkaido", "Tohoku", "Kanto", "Chubu", "Kansai", "Chugoku", "Shikoku", "Kyushu-Okinawa",
    ]

    for region in regions:
        with driver.session() as session:
            session.run("MERGE (r:Region {name: $name})", name=region)
    print(f" Inserite {len(regions)} regioni.")

    # ==========================================
    # 3. SEED: PREFETTURE (47)
    # ==========================================
    prefectures = [
        {"name": "Hokkaido", "region": "Hokkaido"},
        {"name": "Aomori", "region": "Tohoku"}, {"name": "Iwate", "region": "Tohoku"},
        {"name": "Miyagi", "region": "Tohoku"}, {"name": "Akita", "region": "Tohoku"},
        {"name": "Yamagata", "region": "Tohoku"}, {"name": "Fukushima", "region": "Tohoku"},
        {"name": "Ibaraki", "region": "Kanto"}, {"name": "Tochigi", "region": "Kanto"},
        {"name": "Gunma", "region": "Kanto"}, {"name": "Saitama", "region": "Kanto"},
        {"name": "Chiba", "region": "Kanto"}, {"name": "Tokyo", "region": "Kanto"},
        {"name": "Kanagawa", "region": "Kanto"}, {"name": "Niigata", "region": "Chubu"},
        {"name": "Toyama", "region": "Chubu"}, {"name": "Ishikawa", "region": "Chubu"},
        {"name": "Fukui", "region": "Chubu"}, {"name": "Yamanashi", "region": "Chubu"},
        {"name": "Nagano", "region": "Chubu"}, {"name": "Gifu", "region": "Chubu"},
        {"name": "Shizuoka", "region": "Chubu"}, {"name": "Aichi", "region": "Chubu"},
        {"name": "Mie", "region": "Kansai"}, {"name": "Shiga", "region": "Kansai"},
        {"name": "Kyoto", "region": "Kansai"}, {"name": "Osaka", "region": "Kansai"},
        {"name": "Hyogo", "region": "Kansai"}, {"name": "Nara", "region": "Kansai"},
        {"name": "Wakayama", "region": "Kansai"}, {"name": "Tottori", "region": "Chugoku"},
        {"name": "Shimane", "region": "Chugoku"}, {"name": "Okayama", "region": "Chugoku"},
        {"name": "Hiroshima", "region": "Chugoku"}, {"name": "Yamaguchi", "region": "Chugoku"},
        {"name": "Tokushima", "region": "Shikoku"}, {"name": "Kagawa", "region": "Shikoku"},
        {"name": "Ehime", "region": "Shikoku"}, {"name": "Kochi", "region": "Shikoku"},
        {"name": "Fukuoka", "region": "Kyushu-Okinawa"}, {"name": "Saga", "region": "Kyushu-Okinawa"},
        {"name": "Nagasaki", "region": "Kyushu-Okinawa"}, {"name": "Kumamoto", "region": "Kyushu-Okinawa"},
        {"name": "Oita", "region": "Kyushu-Okinawa"}, {"name": "Miyazaki", "region": "Kyushu-Okinawa"},
        {"name": "Kagoshima", "region": "Kyushu-Okinawa"}, {"name": "Okinawa", "region": "Kyushu-Okinawa"},
    ]

    for pref in prefectures:
        with driver.session() as session:
            session.run(
                """
                MERGE (p:Prefecture {name: $name})
                WITH p
                MATCH (r:Region {name: $region})
                MERGE (p)-[:IN_REGION]->(r)
                """,
                name=pref["name"],
                region=pref["region"],
            )
    print(f"Inserite {len(prefectures)} prefetture con relazioni IN_REGION.")

    # ==========================================
    # 4. SEED: CATEGORIE FISSE
    # ==========================================
    categories = [
        "Shrine", "Temple", "Castle", "Garden/Park", "Museum", 
        "Nature/Hike", "Beach", "Modern/City", "Viewpoint", 
        "Onsen", "Historic Site", "Market", "Shopping", "Food",
    ]

    for cat in categories:
        with driver.session() as session:
            session.run("MERGE (c:Category {name: $name})", name=cat)
    print(f"Inserite {len(categories)} categorie.")

    # ==========================================
    # 5. SEED: SPOT INIZIALI CON GERARCHIA A 3 LIVELLI
    # ==========================================
    spots = [
        {"name": "Senso-ji", "lat": 35.7148, "lng": 139.7967, "prefecture": "Tokyo", "city": "Tokyo", "category": "Temple"},
        {"name": "Tokyo Skytree", "lat": 35.7100, "lng": 139.8107, "prefecture": "Tokyo", "city": "Tokyo", "category": "Viewpoint"},
        {"name": "Meiji Jingu", "lat": 35.6764, "lng": 139.6993, "prefecture": "Tokyo", "city": "Tokyo", "category": "Shrine"},
        {"name": "Shibuya Crossing", "lat": 35.6595, "lng": 139.7004, "prefecture": "Tokyo", "city": "Tokyo", "category": "Modern/City"},
        {"name": "Kinkaku-ji", "lat": 35.0394, "lng": 135.7292, "prefecture": "Kyoto", "city": "Kyoto", "category": "Temple"},
        {"name": "Fushimi Inari Taisha", "lat": 34.9671, "lng": 135.7727, "prefecture": "Kyoto", "city": "Kyoto", "category": "Shrine"},
        {"name": "Kiyomizu-dera", "lat": 34.9949, "lng": 135.7850, "prefecture": "Kyoto", "city": "Kyoto", "category": "Temple"},
        {"name": "Arashiyama Bamboo Forest", "lat": 35.0094, "lng": 135.6642, "prefecture": "Kyoto", "city": "Kyoto", "category": "Nature/Hike"},
        {"name": "Gion", "lat": 35.0036, "lng": 135.7750, "prefecture": "Kyoto", "city": "Kyoto", "category": "Historic Site"},
        {"name": "Todai-ji", "lat": 34.6883, "lng": 135.8397, "prefecture": "Nara", "city": "Nara", "category": "Temple"},
        {"name": "Parco di Nara", "lat": 34.6851, "lng": 135.8410, "prefecture": "Nara", "city": "Nara", "category": "Garden/Park"},
        {"name": "Castello di Osaka", "lat": 34.6873, "lng": 135.5262, "prefecture": "Osaka", "city": "Osaka", "category": "Castle"},
        {"name": "Dotonbori", "lat": 34.6686, "lng": 135.5016, "prefecture": "Osaka", "city": "Osaka", "category": "Food"},
        {"name": "Parco della Pace di Hiroshima", "lat": 34.3955, "lng": 132.4533, "prefecture": "Hiroshima", "city": "Hiroshima", "category": "Historic Site"},
        {"name": "Itsukushima Shrine", "lat": 34.2960, "lng": 132.3199, "prefecture": "Hiroshima", "city": "Hatsukaichi", "category": "Shrine"},
        {"name": "Giardino Kenroku-en", "lat": 36.5623, "lng": 136.6624, "prefecture": "Ishikawa", "city": "Kanazawa", "category": "Garden/Park"},
        {"name": "Tempio Toshogu", "lat": 36.7580, "lng": 139.6019, "prefecture": "Tochigi", "city": "Nikko", "category": "Temple"},
        {"name": "Koya-san", "lat": 34.2129, "lng": 135.5860, "prefecture": "Wakayama", "city": "Koya", "category": "Temple"},
        {"name": "Grande Buddha di Kamakura", "lat": 35.3167, "lng": 139.5467, "prefecture": "Kanagawa", "city": "Kamakura", "category": "Historic Site"},
        {"name": "Takayama Jinya", "lat": 36.1411, "lng": 137.2587, "prefecture": "Gifu", "city": "Takayama", "category": "Historic Site"},
    ]

    for spot in spots:
        with driver.session() as session:
            session.run(
                """
                MERGE (s:Spot {name: $name})
                SET s.lat = $lat,
                    s.lng = $lng
                WITH s
                MATCH (p:Prefecture {name: $prefecture})
                
                // Se c'è una città, crea Spot -> City -> Prefecture
                FOREACH (city_name IN CASE WHEN $city IS NOT NULL THEN [$city] ELSE [] END |
                    MERGE (cy:City {name: city_name})
                    MERGE (s)-[:IN_CITY]->(cy)
                    MERGE (cy)-[:IN_PREFECTURE]->(p)
                )
                
                // Se non c'è una città, crea Spot -> Prefecture
                FOREACH (_ IN CASE WHEN $city IS NULL THEN [1] ELSE [] END |
                    MERGE (s)-[:IN_PREFECTURE]->(p)
                )
                
                WITH s
                MATCH (c:Category {name: $category})
                MERGE (s)-[:OF_CATEGORY]->(c)
                RETURN s.name
                """,
                name=spot["name"],
                lat=spot["lat"],
                lng=spot["lng"],
                prefecture=spot["prefecture"],
                city=spot.get("city"),
                category=spot["category"],
            )
    print(f" Inseriti {len(spots)} spot iniziali con struttura gerarchica.")

    # ==========================================
    # 6. SEED: RELAZIONI NEAR
    # ==========================================
    nearby = [
        ("Senso-ji", "Tokyo Skytree"), ("Senso-ji", "Meiji Jingu"),
        ("Meiji Jingu", "Shibuya Crossing"), ("Kinkaku-ji", "Arashiyama Bamboo Forest"),
        ("Kinkaku-ji", "Gion"), ("Fushimi Inari Taisha", "Kiyomizu-dera"),
        ("Kiyomizu-dera", "Gion"), ("Todai-ji", "Parco di Nara"),
        ("Itsukushima Shrine", "Parco della Pace di Hiroshima"),
        ("Castello di Osaka", "Dotonbori"), ("Takayama Jinya", "Giardino Kenroku-en"),
    ]

    for a, b in nearby:
        with driver.session() as session:
            session.run(
                """
                MATCH (a:Spot {name: $a}), (b:Spot {name: $b})
                MERGE (a)-[:NEAR]->(b)
                MERGE (b)-[:NEAR]->(a)
                """,
                a=a, b=b,
            )
    print(f" Create {len(nearby)} relazioni NEAR.")

    

    # ==========================================
    # 8. SEED: FESTIVAL
    # ==========================================
    festivals = [
        {"name": "Gion Matsuri", "month": 7, "prefecture": "Kyoto", "city": "Kyoto"},
        {"name": "Sapporo Snow Festival", "month": 2, "prefecture": "Hokkaido", "city": "Sapporo"},
        {"name": "Hanami (Fioritura dei Ciliegi)", "month": 3, "prefecture": "Tokyo", "city": "Tokyo"},
        {"name": "Sumidagawa Fireworks Festival", "month": 7, "prefecture": "Tokyo", "city": "Tokyo"},
        {"name": "Kanda Matsuri", "month": 5, "prefecture": "Tokyo", "city": "Tokyo"},
        {"name": "Aoi Matsuri", "month": 5, "prefecture": "Kyoto", "city": "Kyoto"},
        {"name": "Tenjin Matsuri", "month": 7, "prefecture": "Osaka", "city": "Osaka"},
        {"name": "Nara Tokae", "month": 8, "prefecture": "Nara", "city": "Nara"},
        {"name": "Nagasaki Kunchi", "month": 10, "prefecture": "Nagasaki", "city": "Nagasaki"},
    ]

    for fest in festivals:
        with driver.session() as session:
            session.run(
                """
                MERGE (f:Festival {name: $name})
                SET f.month = $month
                WITH f
                MATCH (p:Prefecture {name: $prefecture})
                
                FOREACH (city_name IN CASE WHEN $city IS NOT NULL THEN [$city] ELSE [] END |
                    MERGE (cy:City {name: city_name})
                    MERGE (f)-[:HELD_IN]->(cy)
                    MERGE (cy)-[:IN_PREFECTURE]->(p)
                )
                
                FOREACH (_ IN CASE WHEN $city IS NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:HELD_IN]->(p)
                )
                RETURN f.name
                """,
                name=fest["name"],
                month=fest["month"],
                prefecture=fest["prefecture"],
                city=fest.get("city"),
            )
    print(f" Inseriti {len(festivals)} festival.")



    with driver.session() as session:
        result = session.run("MATCH (r:Region) RETURN count(r) AS count")
        print(f" Regioni: {result.single()['count']}")

        result = session.run("MATCH (p:Prefecture) RETURN count(p) AS count")
        print(f"Prefetture: {result.single()['count']}")
        
        # Aggiornato per riflettere il nuovo nodo City
        result = session.run("MATCH (c:City) RETURN count(c) AS count")
        print(f" Città: {result.single()['count']}")

        result = session.run("MATCH (s:Spot) RETURN count(s) AS count")
        print(f" Spot: {result.single()['count']}")

        result = session.run("MATCH (c:Category) RETURN count(c) AS count")
        print(f" Categorie: {result.single()['count']}")

        result = session.run("MATCH (f:Festival) RETURN count(f) AS count")
        print(f" Festival: {result.single()['count']}")

    driver.close()
    print(" Database Neo4j pronto e aggiornato")

if __name__ == "__main__":
    init_ontology()