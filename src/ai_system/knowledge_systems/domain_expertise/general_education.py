"""General Education module — AI Holographic Wristwatch."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger
from src.ai_system.knowledge_systems.domain_expertise.health_knowledge import KnowledgeResult

logger = get_logger(__name__)


class Subject(Enum):
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    LITERATURE = "literature"
    ART = "art"
    PHILOSOPHY = "philosophy"
    ECONOMICS = "economics"


class GeneralEducationBase:
    """Provides general educational knowledge across multiple subjects."""

    DEFINITIONS: Dict[str, str] = {
        # Mathematics
        "integer": "A whole number (not a fraction) that can be positive, negative, or zero: ..., -3, -2, -1, 0, 1, 2, 3, ...",
        "prime number": "A natural number greater than 1 with no positive divisors other than 1 and itself (e.g., 2, 3, 5, 7, 11).",
        "fraction": "A number expressed as one integer divided by another, representing a part of a whole (e.g., 3/4).",
        "percentage": "A ratio or fraction expressed as a part of 100 (e.g., 75% = 75/100 = 0.75).",
        "probability": "A measure of the likelihood of an event occurring, expressed as a number between 0 (impossible) and 1 (certain).",
        "mean": "The arithmetic average of a set of numbers, calculated by summing all values and dividing by the count.",
        "median": "The middle value in a sorted list of numbers. If even count, average of two middle values.",
        "mode": "The value that appears most frequently in a data set.",
        # Science
        "atom": "The smallest unit of a chemical element that retains the properties of that element. Composed of protons, neutrons, and electrons.",
        "molecule": "Two or more atoms bonded together, forming the smallest unit of a chemical compound (e.g., H₂O, CO₂).",
        "evolution": "The process of gradual change in inherited characteristics of biological populations over successive generations, driven by natural selection.",
        "cell": "The basic structural and functional unit of all living organisms. Prokaryotic cells lack a nucleus; eukaryotic cells have one.",
        "ecosystem": "A community of living organisms interacting with their non-living environment (air, water, minerals) in a specific area.",
        "photon": "A quantum of electromagnetic radiation (light) with no mass, traveling at the speed of light.",
        "isotope": "Atoms of the same element with the same number of protons but different numbers of neutrons, giving different mass numbers.",
        # History
        "renaissance": "European cultural, artistic, political, and scientific rebirth (14th-17th centuries), originating in Italy. Marked by renewed interest in classical antiquity.",
        "industrial revolution": "Period of major industrialization from the 18th-19th centuries, beginning in Britain. Transitioned economies from agrarian to manufacturing-based.",
        "world war ii": "Global war (1939-1945) involving most of the world's nations, between the Allied powers and Axis powers (Nazi Germany, Italy, Japan). ~70-85 million people died.",
        "cold war": "Geopolitical tension (1947-1991) between the United States and Soviet Union. Characterized by proxy wars, arms race, and ideological competition, without direct armed conflict.",
        "democracy": "A system of government in which power is vested in the people, who exercise it directly or through elected representatives.",
        "colonialism": "The practice of a country acquiring control over another territory, exploiting it economically and politically, often by settlement.",
        # Geography
        "latitude": "The angular distance of a place north or south of the equator, measured in degrees (0° at equator, ±90° at poles).",
        "longitude": "The angular distance of a place east or west of the prime meridian (Greenwich), measured in degrees (0° to ±180°).",
        "continent": "One of Earth's seven large landmasses: Africa, Antarctica, Asia, Australia, Europe, North America, South America.",
        "biome": "A large naturally occurring community of flora and fauna occupying a major habitat (e.g., tropical rainforest, tundra, desert).",
        "tectonic plates": "Massive segments of Earth's lithosphere that move, float, and sometimes fracture, causing earthquakes, volcanoes, and continental drift.",
        "river delta": "A landform created by deposition of sediment carried by a river as it enters slower-moving or standing water, forming a fan-like shape.",
        # Literature
        "metaphor": "A figure of speech applying a word or phrase to something it does not literally denote, suggesting a resemblance (e.g., 'Life is a journey').",
        "simile": "A figure of speech comparing two different things using 'like' or 'as' (e.g., 'brave as a lion').",
        "irony": "A rhetorical device where the intended meaning is opposite to the literal meaning, or where events differ from expectations.",
        "protagonist": "The leading character or central figure in a drama, movie, novel, or other narrative.",
        "allegory": "A narrative in which characters and events represent abstract ideas or moral qualities, with hidden symbolic meaning.",
        # Art
        "impressionism": "Late 19th-century French art movement characterized by small brush strokes, light colors, and capturing moments in time (Monet, Renoir).",
        "perspective": "Artistic technique creating the illusion of depth and space on a flat surface by depicting objects as appearing smaller at greater distances.",
        "chiaroscuro": "Artistic technique using strong contrasts between light and dark to model three-dimensional objects on a flat surface (Rembrandt, Caravaggio).",
        # Philosophy
        "empiricism": "Philosophical theory that knowledge comes primarily from sensory experience, not innate ideas (Locke, Hume, Bacon).",
        "rationalism": "Philosophical doctrine that reason is the primary source of knowledge, independent of sensory experience (Descartes, Spinoza, Leibniz).",
        "ethics": "Branch of philosophy dealing with questions of moral value: What is good? What is right? How should one live?",
        "epistemology": "Branch of philosophy concerned with the nature, scope, and limits of human knowledge and justified belief.",
        # Economics
        "gdp": "Gross Domestic Product: the total monetary value of all goods and services produced within a country in a specific period.",
        "inflation": "A sustained increase in the general price level of goods and services, decreasing the purchasing power of money.",
        "supply and demand": "Economic model describing the relationship between availability of a product (supply) and desire for it (demand), determining its price.",
        "opportunity cost": "The value of the next-best alternative forgone when making a decision; the hidden cost of choosing one option over another.",
        "monopoly": "A market structure where a single seller dominates the market for a product or service with no close substitutes, controlling price and supply.",
    }

    TOPIC_EXPLANATIONS: Dict[str, Dict] = {
        "the french revolution": {
            "simple": "The French Revolution was a period when French people overthrew their king and completely changed their government.",
            "standard": (
                "The French Revolution (1789-1799) was a period of radical political and societal transformation in France. "
                "Triggered by financial crisis, food shortages, and Enlightenment ideas about liberty and equality, "
                "the Revolution abolished the monarchy, executed King Louis XVI, and declared a republic. "
                "It culminated in Napoleon Bonaparte's rise to power."
            ),
            "detailed": (
                "The French Revolution (1789-1799) emerged from a confluence of factors: fiscal crisis from wars and royal spending, "
                "social inequality (the Third Estate bearing tax burdens while nobility were exempt), Enlightenment philosophies "
                "challenging divine right, and food shortages. Key events: the storming of the Bastille (1789), the Declaration of the "
                "Rights of Man, the Reign of Terror under Robespierre (1793-94) with ~17,000 executions, the Thermidorian Reaction, "
                "and the Directory period. The Revolution dismantled feudalism, established secular governance, and profoundly influenced "
                "subsequent democratic movements worldwide."
            ),
        },
        "climate change": {
            "simple": "Climate change means Earth is getting warmer because of greenhouse gases humans release into the air, causing problems for animals, plants, and people.",
            "standard": (
                "Climate change refers to long-term shifts in global temperatures and weather patterns. "
                "Since the industrial revolution, human activities — burning fossil fuels, deforestation, industrial processes — "
                "have increased greenhouse gases (CO₂, methane, nitrous oxide), trapping heat in the atmosphere. "
                "Consequences include rising sea levels, more extreme weather events, ecosystem disruption, and threats to food security."
            ),
            "detailed": (
                "Anthropogenic climate change is driven by enhanced greenhouse effect from emissions primarily of CO₂ (from fossil fuels), "
                "CH₄ (agriculture and waste), N₂O (fertilizers), and F-gases. The IPCC projects 1.5-4.5°C warming by 2100 under "
                "different emission scenarios. Consequences include: sea level rise (0.3-1m by 2100), increased frequency/intensity of "
                "hurricanes, droughts, and floods; ocean acidification; biodiversity loss; and disrupted agricultural systems. "
                "Mitigation strategies: decarbonizing energy (renewables, nuclear), electrification of transport and heating, "
                "carbon capture, and sustainable land use. Adaptation includes flood defenses, drought-resistant crops, and heat action plans."
            ),
        },
        "world war ii": {
            "simple": "World War II was a huge global war from 1939 to 1945 where the Allied countries fought against Nazi Germany, Italy, and Japan.",
            "standard": (
                "World War II (1939-1945) was the deadliest conflict in human history, involving 30+ countries. "
                "It began with Nazi Germany's invasion of Poland (September 1, 1939). "
                "The war featured the Holocaust (systematic genocide of 6 million Jews), atomic bombings of Hiroshima and Nagasaki, "
                "and resulted in ~70-85 million deaths. The war ended in Europe on May 8, 1945 (VE Day) and in the Pacific on September 2, 1945 (VJ Day)."
            ),
            "detailed": (
                "WWII emerged from unresolved WWI tensions, the Great Depression, rise of fascism (Hitler's Nazi Germany, Mussolini's Italy), "
                "Japanese imperialism, and failed appeasement policies. Key theaters: European (Blitzkrieg, Battle of Britain, Operation Barbarossa, "
                "D-Day), Pacific (Pearl Harbor, island-hopping, Battle of Midway), North African, and Eastern Front. "
                "Holocaust: systematic murder of 6 million Jews and millions of others by Nazis. Post-war restructuring led to the "
                "United Nations, NATO, Marshall Plan, Nuremberg Trials, Cold War, decolonization movements, and modern international law."
            ),
        },
        "evolution": {
            "simple": "Evolution is how living things slowly change over many generations. Animals and plants that are better adapted to their environment survive and have more babies.",
            "standard": (
                "Evolution is the process by which species change over time through natural selection, genetic mutation, and genetic drift. "
                "Charles Darwin proposed natural selection: organisms with favorable traits survive and reproduce more, "
                "passing those traits to offspring. Over millions of years, this leads to new species. "
                "Evidence includes the fossil record, comparative anatomy, genetic similarities, and direct observation of evolution."
            ),
            "detailed": (
                "Modern evolutionary theory (the Modern Synthesis) integrates Darwin's natural selection with Mendelian genetics and "
                "molecular biology. Key mechanisms: natural selection (differential reproductive success based on heritable traits), "
                "genetic drift (random allele frequency changes in small populations), gene flow (allele transfer between populations), "
                "and mutation (source of new genetic variation). Speciation occurs through allopatric (geographic isolation) or "
                "sympatric (ecological differentiation) processes. Molecular evidence: shared DNA sequences, conserved genes (HOX genes), "
                "endogenous retroviruses. Evolution operates on the genetic level (allele frequencies) and is evidenced across "
                "fossil records, comparative genomics, antibiotic resistance, and artificial selection."
            ),
        },
        "supply and demand": {
            "simple": "When there's more of something available, its price usually goes down. When everyone wants something that's hard to find, the price goes up.",
            "standard": (
                "Supply and demand is the foundation of market economics. Demand: as price falls, quantity demanded rises (inverse relationship). "
                "Supply: as price rises, quantity supplied increases (direct relationship). "
                "The equilibrium price is where supply equals demand. Shifts in curves occur due to income changes, "
                "substitute goods, production costs, and consumer preferences."
            ),
            "detailed": (
                "Price theory's core model combines demand functions (consumer utility maximization subject to budget constraints) "
                "and supply functions (producer profit maximization given production costs). "
                "Demand curve shifters: income (normal/inferior goods), prices of complements and substitutes, consumer expectations, "
                "tastes, and population. Supply curve shifters: input costs, technology, number of producers, and expectations. "
                "Market failure occurs with externalities (Pigouvian taxes/subsidies), public goods (non-excludable, non-rival), "
                "information asymmetry (Akerlof's lemons), and monopoly power. Elasticity measures responsiveness: "
                "PED = %ΔQD / %ΔP; PES = %ΔQS / %ΔP. Cross-price and income elasticity quantify substitution and income effects."
            ),
        },
        "democracy": {
            "simple": "Democracy is when people vote for their leaders and get a say in how their country is run.",
            "standard": (
                "Democracy is a system of government where power rests with the people, exercised through regular free elections. "
                "In direct democracy, citizens vote on every issue. In representative democracy, citizens elect representatives. "
                "Core principles include rule of law, separation of powers, protection of individual rights, free press, "
                "and peaceful transfer of power."
            ),
            "detailed": (
                "Democracy's theoretical foundations trace from Athenian direct democracy through Locke's social contract, Montesquieu's "
                "separation of powers, and Rousseau's general will. Modern liberal democracy combines popular sovereignty with "
                "constitutional constraints (counter-majoritarian protections for minorities). Key features: free and fair elections, "
                "multiparty competition, civil liberties, independent judiciary, free press, and civilian control of the military. "
                "Variants include parliamentary (UK), presidential (USA), and semi-presidential (France) systems. "
                "Democratic backsliding (Hungary, Turkey) and illiberal democracy present contemporary challenges to democratic theory."
            ),
        },
    }

    RELATED_TOPICS: Dict[str, List[str]] = {
        "mathematics": ["calculus", "algebra", "geometry", "statistics", "number theory"],
        "science": ["physics", "chemistry", "biology", "astronomy", "earth science"],
        "history": ["archaeology", "political science", "sociology", "cultural studies"],
        "geography": ["geology", "climatology", "oceanography", "cartography"],
        "literature": ["poetry", "drama", "narrative theory", "linguistics", "rhetoric"],
        "art": ["music", "architecture", "sculpture", "photography", "design"],
        "philosophy": ["ethics", "logic", "metaphysics", "epistemology", "aesthetics"],
        "economics": ["microeconomics", "macroeconomics", "finance", "political economy", "sociology"],
        "the french revolution": ["Napoleon Bonaparte", "Enlightenment", "European history", "democracy"],
        "climate change": ["greenhouse effect", "renewable energy", "fossil fuels", "ecology"],
        "world war ii": ["Holocaust", "Cold War", "decolonization", "United Nations"],
        "evolution": ["genetics", "paleontology", "natural selection", "Charles Darwin"],
        "supply and demand": ["market equilibrium", "price theory", "elasticity", "microeconomics"],
        "democracy": ["human rights", "elections", "separation of powers", "civil liberties"],
        "prime number": ["number theory", "cryptography", "factorization", "Riemann hypothesis"],
        "atom": ["chemistry", "quantum mechanics", "periodic table", "nuclear physics"],
        "renaissance": ["art history", "humanism", "Medici family", "Leonardo da Vinci"],
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._query_count: int = 0
        logger.info("GeneralEducationBase initialized")

    def query(self, question: str) -> KnowledgeResult:
        """Search for educational content using keyword matching."""
        with self._lock:
            self._query_count += 1

        q_lower = question.lower().strip()

        # Check definitions
        if q_lower in self.DEFINITIONS:
            return KnowledgeResult(
                query=question,
                answer=self.DEFINITIONS[q_lower],
                confidence=0.95,
                sources=["Encyclopedia Britannica", "Oxford Dictionary"],
                related_topics=self.get_related_topics(q_lower),
            )

        # Check topic explanations
        if q_lower in self.TOPIC_EXPLANATIONS:
            explanation = self.TOPIC_EXPLANATIONS[q_lower].get("standard", "")
            return KnowledgeResult(
                query=question,
                answer=explanation,
                confidence=0.92,
                sources=["Encyclopedia Britannica", "Wikipedia"],
                related_topics=self.get_related_topics(q_lower),
            )

        # Keyword fuzzy search across definitions
        best_key: Optional[str] = None
        best_score: float = 0.0
        q_words = set(q_lower.split())

        for key in list(self.DEFINITIONS.keys()) + list(self.TOPIC_EXPLANATIONS.keys()):
            key_words = set(key.split())
            overlap = len(q_words & key_words)
            score = overlap / max(len(q_words), len(key_words), 1)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= 0.3:
            if best_key in self.DEFINITIONS:
                answer = self.DEFINITIONS[best_key]
            else:
                answer = self.TOPIC_EXPLANATIONS[best_key].get("standard", "")
            return KnowledgeResult(
                query=question,
                answer=answer,
                confidence=min(0.88, best_score * 1.2),
                sources=["Encyclopedia Britannica", "Wikipedia"],
                related_topics=self.get_related_topics(best_key),
            )

        return KnowledgeResult(
            query=question,
            answer=f"I don't have specific educational content for '{question}'. Please consult educational resources for detailed information.",
            confidence=0.1,
            sources=[],
            related_topics=[],
        )

    def get_definition(self, term: str) -> Optional[str]:
        """Return definition for a given term."""
        term_lower = term.lower().strip()
        if term_lower in self.DEFINITIONS:
            return self.DEFINITIONS[term_lower]
        # Partial match
        for key, definition in self.DEFINITIONS.items():
            if term_lower in key or key in term_lower:
                return definition
        return None

    def explain_topic(self, topic: str, depth: int = 2) -> str:
        """
        Explain a topic at a specified depth level.
        1 = simple, 3 = standard, 5 = expert-level detailed.
        """
        topic_lower = topic.lower().strip()
        depth = max(1, min(5, depth))

        explanation_entry: Optional[Dict] = None

        # Direct lookup
        if topic_lower in self.TOPIC_EXPLANATIONS:
            explanation_entry = self.TOPIC_EXPLANATIONS[topic_lower]
        else:
            # Fuzzy match
            for key, entry in self.TOPIC_EXPLANATIONS.items():
                if any(word in topic_lower for word in key.split() if len(word) > 4):
                    explanation_entry = entry
                    break

        if explanation_entry is None:
            # Fall back to definition
            definition = self.get_definition(topic)
            if definition:
                return definition
            return f"No explanation available for '{topic}' at this time."

        if depth <= 1:
            return explanation_entry.get("simple", explanation_entry.get("standard", ""))
        elif depth <= 3:
            return explanation_entry.get("standard", explanation_entry.get("simple", ""))
        else:
            return explanation_entry.get("detailed", explanation_entry.get("standard", ""))

    def get_related_topics(self, topic: str) -> List[str]:
        """Return list of related topics."""
        topic_lower = topic.lower().strip()
        if topic_lower in self.RELATED_TOPICS:
            return self.RELATED_TOPICS[topic_lower]
        # Fuzzy match
        for key, related in self.RELATED_TOPICS.items():
            if key in topic_lower or topic_lower in key:
                return related
        return ["education", "knowledge", "learning"]

    def get_stats(self) -> Dict:
        """Return operational statistics."""
        with self._lock:
            return {
                "query_count": self._query_count,
                "total_definitions": len(self.DEFINITIONS),
                "total_topic_explanations": len(self.TOPIC_EXPLANATIONS),
                "subjects": [s.value for s in Subject],
            }


_general_education_instance: Optional[GeneralEducationBase] = None
_general_education_lock = threading.Lock()


def get_general_education_base() -> GeneralEducationBase:
    """Return the singleton GeneralEducationBase instance."""
    global _general_education_instance
    if _general_education_instance is None:
        with _general_education_lock:
            if _general_education_instance is None:
                _general_education_instance = GeneralEducationBase()
    return _general_education_instance
