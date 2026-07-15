import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class Neo4jKGManager:
    def __init__(self):
        URI = os.getenv("NEO4J_URI")
        AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        self.driver = GraphDatabase.driver(URI, auth=AUTH)

    def close(self):
        self.driver.close()


    def get_recent_posts(self, limit: int = 5) -> list:
        """
        Recupera gli ultimi post generati per evitare ripetizioni nel planner.
        """
        query = """
        MATCH (b:BlogPost)
        RETURN b.title AS title, b.type AS type, b.base_topic AS topic, b.angle AS angle, b.created_at AS date
        ORDER BY b.created_at DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]
        
    def get_coverage_gaps(self):
        """
        Restituisce le prefetture ordinate per numero di post crescente
        (gap di copertura).
        """
        query = """
        MATCH (p:Prefecture)
        OPTIONAL MATCH (p)<-[:COVERS_PREFECTURE]-(b:BlogPost)
        RETURN p.name AS prefecture, count(b) AS coverage_count
        ORDER BY coverage_count ASC LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def get_upcoming_festivals(self, month: int = None) -> list:
        """
        Recupera i festival per un mese specifico (o il mese corrente).
        """
        if month is None:
            from datetime import datetime
            month = datetime.now().month

        query = """
        MATCH (f:Festival {month: $month})
        OPTIONAL MATCH (f)-[:HELD_IN]->(p:Prefecture)
        RETURN f.name AS name, f.month AS month, collect(DISTINCT p.name) AS prefectures
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, month=month)
            return [record.data() for record in result]

    def get_theme_coverage_gaps(self, limit: int = 5) -> list:
        """
        Recupera le categorie ordinate per numero di post crescente.
        """
        query = """
        MATCH (c:Category)
        OPTIONAL MATCH (c)<-[:OF_CATEGORY]-(s:Spot)<-[:ABOUT]-(b:BlogPost)
        RETURN c.name AS category, count(DISTINCT b) AS post_count
        ORDER BY post_count ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def get_entities_for_krag(self, entity_name: str) -> dict:
        """
        Estrae informazioni per un'entità, cercando in ordine: come Spot, come City e infine come Prefecture.
        Restituisce un dizionario con target_spot, nearby_spots, categories, prefecture_location.
        """
        # 1. Prova come Spot (Attrazione)
        query_spot = """
        MATCH (s:Spot {name: $name})
        OPTIONAL MATCH (s)-[:NEAR]-(near:Spot)
        OPTIONAL MATCH (s)-[:OF_CATEGORY]->(c:Category)
        OPTIONAL MATCH (s)-[:IN_CITY]->(cy:City)
        OPTIONAL MATCH (s)-[:IN_PREFECTURE]->(p:Prefecture)
        RETURN s.name AS target_spot,
            collect(DISTINCT near.name) AS nearby_spots,
            collect(DISTINCT c.name) AS categories,
            collect(DISTINCT cy.name) + collect(DISTINCT p.name) AS prefecture_location
        """
        with self.driver.session() as session:
            result = session.run(query_spot, name=entity_name)
            record = result.single()
            if record:
                return record.data()

            # 2.Se non è uno Spot, prova come City (Città)
            query_city = """
            MATCH (cy:City {name: $name})
            OPTIONAL MATCH (s:Spot)-[:IN_CITY]->(cy)
            OPTIONAL MATCH (cy)-[:IN_PREFECTURE]->(p:Prefecture)
            // Se stiamo cercando una città, consideriamo gli Spot al suo interno come 'nearby_spots'
            RETURN cy.name AS target_spot,
                collect(DISTINCT s.name) AS nearby_spots, 
                [] AS categories,
                collect(DISTINCT p.name) AS prefecture_location
            """
            result2 = session.run(query_city, name=entity_name)
            record2 = result2.single()
            if record2:
                return record2.data()

            # 3. Se non è City, prova come Prefecture (Prefettura)
            query_pref = """
            MATCH (p:Prefecture {name: $name})
            OPTIONAL MATCH (cy:City)-[:IN_PREFECTURE]->(p)
            RETURN p.name AS target_spot,
                collect(DISTINCT cy.name) AS nearby_spots,
                [] AS categories,
                [p.name] AS prefecture_location
            """
            result3 = session.run(query_pref, name=entity_name)
            record3 = result3.single()
            if record3:
                return record3.data()

            # 4. Se non trovato, prova una ricerca fuzzy (case-insensitive)
            query_fuzzy = """
            MATCH (n)
            WHERE (n:Spot OR n:City OR n:Prefecture) AND toLower(n.name) = toLower($name)
            RETURN n.name AS target_spot,
                [] AS nearby_spots,
                [] AS categories,
                [] AS prefecture_location
            LIMIT 1
            """
            result4 = session.run(query_fuzzy, name=entity_name)
            record4 = result4.single()
            if record4:
                return record4.data()

            return {}

    def check_existing_posts(self, spot_name: str) -> list:
        """
        Verifica se esistono già post su questa attrazione (per nome).
        """
        query = """
        MATCH (s:Spot {name: $name})<-[:ABOUT]-(b:BlogPost)
        RETURN b.title AS title, b.type AS type, b.created_at AS date
        ORDER BY b.created_at DESC
        """
        with self.driver.session() as session:
            result = session.run(query, name=spot_name)
            return [record.data() for record in result]
        
    def query(self, entity_name: str) -> dict:
        """
        Scheda completa per nome.
        """
        entities = self.get_entities_for_krag(entity_name)
        posts = self.check_existing_posts(entity_name)
        return {"krag_context": entities, "editorial_history": posts}
    
    def update_after_approval(self, post_data: dict, extracted_entities: dict):
        query = """
        // 1. Crea o Aggiorna il BlogPost
        MERGE (b:BlogPost {title: $post_title})
        ON CREATE SET 
            b.base_topic = $base_topic,
            b.type = $post_type,
            b.angle = $angle,
            b.created_at = datetime()
        ON MATCH SET 
            b.base_topic = $base_topic,
            b.type = $post_type,
            b.angle = $angle,
            b.updated_at = datetime()

        // 2. GESTIONE GERARCHIA GEOGRAFICA DINAMICA
        WITH b
        CALL(b) {
            UNWIND coalesce($mapped_locations, []) AS loc
            
            // Salta le entry senza una prefettura valida
            WITH loc WHERE loc.prefecture IS NOT NULL AND trim(loc.prefecture) <> ""
            MERGE (p:Prefecture {name: trim(loc.prefecture)})
            MERGE (b)-[:COVERS_PREFECTURE]->(p)
            
            // Condizionale: se c'è la città, collegala alla prefettura corretta
            FOREACH (ignoreMe IN CASE WHEN loc.city IS NOT NULL AND trim(loc.city) <> "" THEN [1] ELSE [] END |
                MERGE (cy:City {name: trim(loc.city)})
                MERGE (cy)-[:IN_PREFECTURE]->(p)
                MERGE (b)-[:COVERS_CITY]->(cy)
                
                // Condizionale: se c'è lo spot, collegalo alla città corretta
                FOREACH (ignoreMe2 IN CASE WHEN loc.spot IS NOT NULL AND trim(loc.spot) <> "" THEN [1] ELSE [] END |
                    MERGE (s:Spot {name: trim(loc.spot)})
                    MERGE (s)-[:IN_CITY]->(cy)
                    MERGE (b)-[:ABOUT]->(s)
                )
            )
            RETURN count(loc) AS processed_locs
        }
        
        // 3. RECUPERO SPOT PER I COLLEGAMENTI SECONDARI
        WITH b
        OPTIONAL MATCH (b)-[:ABOUT]->(s:Spot)
        WITH b, collect(DISTINCT s) as spots
        
        // 4. Collega le Categorie
        WITH b, spots
        CALL(b) {
            UNWIND coalesce($categories, []) AS cat_name
            WITH cat_name WHERE cat_name IS NOT NULL AND cat_name <> ""
            MERGE (c:Category {name: cat_name})
            MERGE (b)-[:OF_CATEGORY]->(c)
            RETURN collect(c) AS cats
        }

        // 5. Collega il Cibo Locale
        WITH b, spots, cats
        CALL(b) {
            UNWIND coalesce($local_foods, []) AS food_name
            WITH  food_name WHERE food_name IS NOT NULL AND food_name <> ""
            MERGE (lf:LocalFood {name: food_name})
            MERGE (b)-[:HAS_LOCAL_FOOD]->(lf)
            RETURN collect(lf) AS foods
        }
        
        // 6. Collega l'Artigianato
        WITH b, spots, cats, foods
        CALL(b) {
            UNWIND coalesce($crafts, []) AS craft_name
            WITH craft_name WHERE craft_name IS NOT NULL AND craft_name <> ""
            MERGE (cr:Craft {name: craft_name})
            MERGE (b)-[:HAS_CRAFT]->(cr)
            RETURN collect(cr) AS crafts_nodes
        }
        
        // 7. Collegamenti incrociati (Cibo/Categorie su Spot)
        WITH b, spots, cats, foods
        FOREACH (s IN spots |
            FOREACH (c IN cats | MERGE (s)-[:OF_CATEGORY]->(c))
            FOREACH (lf IN foods | MERGE (s)-[:OFFERS_FOOD]->(lf))
        )

        // 8. Collegamento di vicinanza tra Spot
        WITH b, spots
        CALL(b) {
            WITH spots
            UNWIND spots AS s1
            UNWIND spots AS s2
            WITH s1, s2 WHERE elementId(s1) < elementId(s2)
            MATCH (s1)-[:IN_CITY|IN_PREFECTURE]->(shared_loc)<-[:IN_CITY|IN_PREFECTURE]-(s2)
            MERGE (s1)-[:NEAR]->(s2)
            MERGE (s2)-[:NEAR]->(s1)
            RETURN count(*) AS near_rels
        }

        // 9. Fonti e Claim
        WITH b
        CALL(b) {
            UNWIND coalesce($sources, []) AS src_url
            WITH src_url WHERE src_url IS NOT NULL AND src_url <> ""
            MERGE (src:Source {url: src_url})
            MERGE (b)-[:USED_SOURCE]->(src)
            RETURN count(src) AS src_count
        }
        WITH b
        CALL(b) {
            UNWIND coalesce($claims, []) AS claim_text
            WITH claim_text WHERE claim_text IS NOT NULL AND claim_text <> ""
            CREATE (cl:Claim {text: claim_text})
            MERGE (b)-[:CLAIMS]->(cl)
            RETURN count(cl) AS cl_count
        }

        RETURN b.title AS saved_post
        """

        params = {
            "mapped_locations": extracted_entities.get("mapped_locations", []),
            "categories": extracted_entities.get("category", []),
            "local_foods": extracted_entities.get("local_foods", []),
            "crafts": extracted_entities.get("crafts", []),
            "sources": extracted_entities.get("sources", []),
            "claims": extracted_entities.get("claims", []),
            "post_title": post_data.get("title", "Senza Titolo"),
            "base_topic": post_data.get("topic", "Sconosciuto"),
            "post_type": post_data.get("type", "guide"),
            "angle": post_data.get("angle", ""),
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                record = result.single()
                return record.data() if record else {"saved_post": "Titolo Sconosciuto"}
        except Exception as e:
            print(f"[KG MANAGER] Errore Cypher: {e}")
            return {"saved_post": "Errore"}

    def save_active_plan(self, sequence: list):
        plan_json = json.dumps(sequence)
        query = """
        MERGE (p:EditorialPlan {id: 'current_plan'})
        SET p.sequence = $plan_json, p.updated_at = datetime(), p.status = 'active'
        """
        with self.driver.session() as session:
            session.run(query, plan_json=plan_json)

    def get_active_plan_status(self):
        query = "MATCH (p:EditorialPlan {id: 'current_plan'}) RETURN p.sequence AS sequence"
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return json.loads(record["sequence"]) if record and record["sequence"] else None


kg_manager = Neo4jKGManager()