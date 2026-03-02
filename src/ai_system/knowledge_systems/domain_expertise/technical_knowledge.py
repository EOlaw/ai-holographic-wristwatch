"""Technical Knowledge module — AI Holographic Wristwatch."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.core.utils.logger import get_logger
from src.ai_system.knowledge_systems.domain_expertise.health_knowledge import KnowledgeResult

logger = get_logger(__name__)


class TechDomain(Enum):
    PROGRAMMING = "programming"
    ELECTRONICS = "electronics"
    NETWORKING = "networking"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    MATHEMATICS = "mathematics"


class TechnicalKnowledgeBase:
    """Provides technical knowledge including programming, science, engineering, and math concepts."""

    # (key, domain, answer)
    KNOWLEDGE_BASE: Dict[str, Dict] = {
        "what is python": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "Python is a high-level, interpreted, dynamically-typed programming language created by Guido van Rossum "
                "in 1991. Known for its clean, readable syntax and versatility, it is used in web development, data science, "
                "AI/ML, automation, scientific computing, and more. Key features include dynamic typing, garbage collection, "
                "an extensive standard library, and support for multiple programming paradigms."
            ),
        },
        "what is object oriented programming": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "Object-Oriented Programming (OOP) is a programming paradigm based on 'objects' — data structures combining "
                "state (attributes) and behavior (methods). Core principles are Encapsulation (bundling data and methods), "
                "Abstraction (hiding implementation details), Inheritance (deriving new classes from existing ones), "
                "and Polymorphism (same interface, different implementations)."
            ),
        },
        "what is a variable in programming": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "A variable is a named storage location that holds a value that can change during program execution. "
                "Variables have a name (identifier), a type (integer, string, etc.), and a value. "
                "In Python: x = 10 creates an integer variable named x with value 10. "
                "Variables allow programs to store, modify, and retrieve data dynamically."
            ),
        },
        "what is a function in programming": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "A function is a reusable block of code that performs a specific task. Functions take input parameters, "
                "execute logic, and optionally return a value. They promote code reuse, modularity, and readability. "
                "In Python: def add(a, b): return a + b defines a function. Functions are fundamental to structured programming."
            ),
        },
        "what is recursion": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down "
                "into smaller sub-problems of the same type. Every recursive function needs a base case to stop recursion. "
                "Classic examples include factorial calculation and tree traversal. "
                "While elegant, recursion can cause stack overflow if not managed carefully."
            ),
        },
        "what is an algorithm": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "An algorithm is a finite, ordered set of instructions or steps that solves a specific problem or accomplishes "
                "a task. Good algorithms are correct, efficient (in time and space), and clearly specified. "
                "Algorithm complexity is measured with Big-O notation. Common algorithm types include sorting (QuickSort, MergeSort), "
                "searching (binary search), and graph traversal (BFS, DFS)."
            ),
        },
        "what is machine learning": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "Machine learning (ML) is a subset of AI that enables computers to learn from data without being explicitly "
                "programmed. Key categories: Supervised learning (labeled data), Unsupervised learning (unlabeled data), "
                "and Reinforcement learning (reward-based). Common algorithms include linear regression, decision trees, "
                "neural networks, and support vector machines."
            ),
        },
        "what is ohms law": {
            "domain": TechDomain.ELECTRONICS,
            "answer": (
                "Ohm's Law states that the current through a conductor is directly proportional to the voltage across it "
                "and inversely proportional to its resistance: V = IR (Voltage = Current × Resistance). "
                "Where V is in volts, I is in amperes, and R is in ohms. "
                "This fundamental law is essential for analyzing electrical circuits."
            ),
        },
        "what is a resistor": {
            "domain": TechDomain.ELECTRONICS,
            "answer": (
                "A resistor is a passive electronic component that opposes the flow of electric current, converting "
                "electrical energy to heat. Resistance is measured in ohms (Ω). Resistors are used to limit current, "
                "divide voltage, bias transistors, and set signal levels. The color code bands on resistors encode their "
                "resistance value and tolerance."
            ),
        },
        "what is a transistor": {
            "domain": TechDomain.ELECTRONICS,
            "answer": (
                "A transistor is a semiconductor device used to amplify or switch electronic signals. "
                "BJTs (Bipolar Junction Transistors) have three regions: emitter, base, and collector. "
                "MOSFETs (Metal-Oxide-Semiconductor Field-Effect Transistors) are the dominant type in digital circuits. "
                "Transistors are the fundamental building blocks of modern electronics and computers."
            ),
        },
        "what is capacitance": {
            "domain": TechDomain.ELECTRONICS,
            "answer": (
                "Capacitance is the ability of a component to store electrical charge. A capacitor stores energy in "
                "an electric field between two conducting plates separated by a dielectric. Capacitance is measured "
                "in farads (F). Capacitors are used for filtering, energy storage, timing circuits, and signal coupling/decoupling."
            ),
        },
        "what is tcp ip": {
            "domain": TechDomain.NETWORKING,
            "answer": (
                "TCP/IP (Transmission Control Protocol/Internet Protocol) is the foundational communication protocol suite "
                "of the internet. IP handles addressing and routing of packets. TCP provides reliable, ordered, and "
                "error-checked delivery of data between applications. The TCP/IP model has 4 layers: "
                "Application, Transport, Internet, and Network Access."
            ),
        },
        "what is dns": {
            "domain": TechDomain.NETWORKING,
            "answer": (
                "DNS (Domain Name System) is the internet's 'phone book.' It translates human-readable domain names "
                "(like www.example.com) into IP addresses (like 93.184.216.34) that computers use to identify each other. "
                "DNS uses a hierarchical structure with root servers, top-level domain servers, and authoritative name servers."
            ),
        },
        "what is http": {
            "domain": TechDomain.NETWORKING,
            "answer": (
                "HTTP (HyperText Transfer Protocol) is the protocol underlying data communication on the web. "
                "It is a request-response protocol between clients and servers. HTTPS is the encrypted version using TLS/SSL. "
                "HTTP methods include GET (retrieve), POST (send data), PUT (update), DELETE (remove), and PATCH (partial update)."
            ),
        },
        "what is an ip address": {
            "domain": TechDomain.NETWORKING,
            "answer": (
                "An IP (Internet Protocol) address is a unique numerical label assigned to each device on a network. "
                "IPv4 uses 32-bit addresses (e.g., 192.168.1.1), providing ~4.3 billion addresses. "
                "IPv6 uses 128-bit addresses (e.g., 2001:0db8::1), providing a virtually unlimited address space. "
                "Private IP ranges (192.168.x.x, 10.x.x.x) are for local networks; public IPs for the internet."
            ),
        },
        "what is newton's first law": {
            "domain": TechDomain.SCIENCE,
            "answer": (
                "Newton's First Law of Motion (Law of Inertia) states that an object at rest stays at rest, and an object "
                "in motion stays in motion at the same speed and direction, unless acted upon by an unbalanced external force. "
                "Inertia is the tendency of objects to resist changes in their state of motion."
            ),
        },
        "what is gravity": {
            "domain": TechDomain.SCIENCE,
            "answer": (
                "Gravity is the fundamental force of attraction between all objects with mass. Newton described it as "
                "F = Gm₁m₂/r² (force proportional to product of masses, inversely proportional to distance squared). "
                "Einstein's General Relativity describes gravity as the curvature of spacetime caused by mass and energy. "
                "On Earth's surface, gravitational acceleration is approximately 9.8 m/s²."
            ),
        },
        "what is photosynthesis": {
            "domain": TechDomain.SCIENCE,
            "answer": (
                "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy (from the sun) "
                "into chemical energy stored as glucose. The overall equation: "
                "6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. "
                "It occurs in chloroplasts and consists of the light-dependent reactions (thylakoids) and "
                "the Calvin cycle (stroma)."
            ),
        },
        "what is the speed of light": {
            "domain": TechDomain.SCIENCE,
            "answer": (
                "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3×10⁸ m/s or 186,282 miles/s). "
                "According to Einstein's special relativity, it is the maximum speed at which information or matter can travel. "
                "It is the constant 'c' in the famous equation E = mc²."
            ),
        },
        "what is thermodynamics": {
            "domain": TechDomain.SCIENCE,
            "answer": (
                "Thermodynamics is the branch of physics dealing with heat, work, temperature, and energy. "
                "The four laws: Zeroth (thermal equilibrium), First (energy conservation: ΔU = Q - W), "
                "Second (entropy of isolated systems increases), Third (entropy approaches constant as temperature approaches absolute zero). "
                "It underpins engines, refrigeration, chemistry, and statistical mechanics."
            ),
        },
        "what is the pythagorean theorem": {
            "domain": TechDomain.MATHEMATICS,
            "answer": (
                "The Pythagorean Theorem states that in a right-angled triangle, the square of the hypotenuse (longest side) "
                "equals the sum of the squares of the other two sides: a² + b² = c², where c is the hypotenuse. "
                "It is one of the most fundamental theorems in Euclidean geometry and has applications in navigation, "
                "construction, physics, and computer graphics."
            ),
        },
        "what is calculus": {
            "domain": TechDomain.MATHEMATICS,
            "answer": (
                "Calculus is the mathematical study of continuous change. It has two main branches: "
                "Differential calculus (rates of change, derivatives, slopes of curves) and "
                "Integral calculus (accumulation, areas under curves). "
                "Developed independently by Newton and Leibniz in the 17th century. "
                "Essential for physics, engineering, economics, and computer science."
            ),
        },
        "what is a derivative": {
            "domain": TechDomain.MATHEMATICS,
            "answer": (
                "A derivative measures the rate of change of a function with respect to a variable. "
                "Geometrically, it is the slope of the tangent line to a curve at a point. "
                "Notation: f'(x), dy/dx, or df/dx. For example, the derivative of x² is 2x. "
                "Derivatives are fundamental to optimization, physics (velocity from position), and machine learning (gradient descent)."
            ),
        },
        "what is an integral": {
            "domain": TechDomain.MATHEMATICS,
            "answer": (
                "An integral is the reverse operation of differentiation and represents the accumulation of quantities. "
                "Definite integrals compute the area under a curve between two bounds. "
                "Indefinite integrals find antiderivatives. The Fundamental Theorem of Calculus connects derivatives and integrals. "
                "Applications include calculating areas, volumes, work done by a force, and probability distributions."
            ),
        },
        "what is a matrix": {
            "domain": TechDomain.MATHEMATICS,
            "answer": (
                "A matrix is a rectangular array of numbers, symbols, or expressions arranged in rows and columns. "
                "An m×n matrix has m rows and n columns. Matrices are used in linear algebra for solving systems of equations, "
                "transformations in computer graphics, neural network weights in ML, and quantum mechanics. "
                "Operations include addition, scalar multiplication, matrix multiplication, transposition, and inversion."
            ),
        },
        "what is binary code": {
            "domain": TechDomain.ENGINEERING,
            "answer": (
                "Binary code is a system representing information using only two symbols: 0 and 1 (bits). "
                "It is the foundation of digital computing since electronic circuits can represent two states (on/off, high/low voltage). "
                "8 bits = 1 byte. Numbers, text, images, and instructions are all encoded in binary. "
                "Decimal 10 = Binary 1010 = Hex A."
            ),
        },
        "what is an api": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "An API (Application Programming Interface) is a set of rules and protocols that allows different software "
                "applications to communicate with each other. It defines how to request services from a library or application. "
                "REST APIs use HTTP and return JSON/XML. APIs enable integration between apps, access to external services, "
                "and modular software design."
            ),
        },
        "what is cloud computing": {
            "domain": TechDomain.ENGINEERING,
            "answer": (
                "Cloud computing is the delivery of computing services (servers, storage, databases, networking, software) "
                "over the internet ('the cloud') on demand. Key service models: IaaS (Infrastructure as a Service), "
                "PaaS (Platform as a Service), SaaS (Software as a Service). Major providers: AWS, Microsoft Azure, Google Cloud. "
                "Benefits include scalability, cost efficiency, and accessibility."
            ),
        },
        "what is encryption": {
            "domain": TechDomain.NETWORKING,
            "answer": (
                "Encryption is the process of converting readable data (plaintext) into an unreadable format (ciphertext) "
                "using an algorithm and a key. Only those with the correct decryption key can access the original data. "
                "Symmetric encryption (AES) uses the same key for encryption/decryption. "
                "Asymmetric encryption (RSA, ECC) uses a public/private key pair. HTTPS uses TLS for encrypted web communication."
            ),
        },
        "what is a database": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "A database is an organized collection of structured data stored electronically. "
                "Relational databases (SQL) store data in tables with defined schemas (MySQL, PostgreSQL). "
                "NoSQL databases store flexible/unstructured data (MongoDB, Redis, Cassandra). "
                "Key operations: CRUD (Create, Read, Update, Delete). Databases are indexed for fast retrieval."
            ),
        },
        "what is version control": {
            "domain": TechDomain.PROGRAMMING,
            "answer": (
                "Version control systems (VCS) track changes to files over time, allowing teams to collaborate and "
                "revert to previous versions. Git is the most popular distributed VCS. Key concepts: repository, commit, "
                "branch, merge, pull request. Git platforms include GitHub, GitLab, and Bitbucket. "
                "Good version control practices include atomic commits, meaningful messages, and regular pushes."
            ),
        },
    }

    HOW_TO_GUIDES: Dict[str, List[str]] = {
        "set up python environment": [
            "1. Download Python from python.org and install it for your OS.",
            "2. Verify installation: run 'python --version' in terminal.",
            "3. Create a virtual environment: 'python -m venv myenv'",
            "4. Activate the virtual environment: On Windows: 'myenv\\Scripts\\activate'; On macOS/Linux: 'source myenv/bin/activate'",
            "5. Install packages with pip: 'pip install package-name'",
            "6. Freeze dependencies: 'pip freeze > requirements.txt'",
            "7. Deactivate: run 'deactivate'",
        ],
        "create a git repository": [
            "1. Install Git from git-scm.com.",
            "2. Configure identity: 'git config --global user.name \"Your Name\"' and 'git config --global user.email \"you@example.com\"'",
            "3. Navigate to your project folder: 'cd /path/to/project'",
            "4. Initialize the repo: 'git init'",
            "5. Stage files: 'git add .' or 'git add filename'",
            "6. Create a commit: 'git commit -m \"Initial commit\"'",
            "7. Add remote: 'git remote add origin https://github.com/user/repo.git'",
            "8. Push: 'git push -u origin main'",
        ],
        "build a simple circuit": [
            "1. Gather components: breadboard, LED, 220Ω resistor, 9V battery, jumper wires.",
            "2. Calculate current: I = (V_supply - V_LED) / R = (9V - 2V) / 220Ω ≈ 32mA (acceptable for most LEDs).",
            "3. Connect the positive terminal of battery to one end of the resistor.",
            "4. Connect the other end of the resistor to the anode (longer leg) of the LED.",
            "5. Connect the cathode (shorter leg) of the LED to the negative terminal of battery.",
            "6. Double-check polarity before connecting power.",
            "7. Connect the battery — the LED should light up.",
        ],
        "set up wifi network": [
            "1. Connect the router to your modem using an Ethernet cable.",
            "2. Power on both devices and wait 2 minutes for them to boot.",
            "3. Connect a computer to the router via Ethernet or default WiFi network.",
            "4. Open a browser and navigate to the router's default gateway (usually 192.168.1.1 or 192.168.0.1).",
            "5. Log in with default credentials (found on the router label).",
            "6. Change the admin password immediately.",
            "7. Go to WiFi settings: set SSID (network name) and choose WPA3 or WPA2 encryption.",
            "8. Set a strong WiFi password (12+ characters, mixed types).",
            "9. Save settings — devices can now connect using the new SSID and password.",
        ],
        "debug a python program": [
            "1. Read the error message carefully — note the exception type and line number.",
            "2. Add print statements or use logging to trace variable values at key points.",
            "3. Use Python's built-in debugger: import pdb; pdb.set_trace() (or 'breakpoint()' in Python 3.7+).",
            "4. Use an IDE debugger (VSCode, PyCharm) to set breakpoints and inspect state.",
            "5. Check assumptions: verify input types, expected vs. actual values.",
            "6. Isolate the problem: create a minimal reproducible example.",
            "7. Search the error message online or consult documentation.",
            "8. Test the fix and write a unit test to prevent regression.",
        ],
        "optimize sql query": [
            "1. Use EXPLAIN or EXPLAIN ANALYZE to see the query execution plan.",
            "2. Add indexes on columns used in WHERE, JOIN, and ORDER BY clauses.",
            "3. Avoid SELECT * — only retrieve the columns you need.",
            "4. Use JOINs instead of subqueries where possible.",
            "5. Limit result sets with LIMIT when not all rows are needed.",
            "6. Avoid functions on indexed columns in WHERE clauses (prevents index use).",
            "7. Consider query caching and connection pooling for high-traffic scenarios.",
            "8. Partition large tables for more efficient queries.",
        ],
        "set up docker container": [
            "1. Install Docker Desktop from docker.com.",
            "2. Create a Dockerfile in your project root.",
            "3. Define the base image: FROM python:3.11-slim",
            "4. Set working directory: WORKDIR /app",
            "5. Copy requirements: COPY requirements.txt .",
            "6. Install dependencies: RUN pip install -r requirements.txt",
            "7. Copy source code: COPY . .",
            "8. Expose port if needed: EXPOSE 8080",
            "9. Define start command: CMD [\"python\", \"main.py\"]",
            "10. Build: 'docker build -t myapp .'",
            "11. Run: 'docker run -p 8080:8080 myapp'",
        ],
        "calculate resistor value": [
            "1. Determine the supply voltage (V_supply).",
            "2. Look up the LED forward voltage (V_f) — typically 1.8-3.3V depending on color.",
            "3. Determine the desired LED current (I_f) — typically 10-20mA.",
            "4. Use Ohm's law: R = (V_supply - V_f) / I_f",
            "5. Example: R = (5V - 2V) / 0.02A = 150Ω",
            "6. Choose the nearest standard resistor value (e.g., 150Ω or 220Ω).",
            "7. Verify power dissipation: P = I² × R. Ensure it is within the resistor's rating.",
        ],
        "implement binary search": [
            "1. Ensure the array is sorted in ascending order.",
            "2. Set low = 0, high = len(array) - 1.",
            "3. While low <= high: calculate mid = (low + high) // 2",
            "4. If array[mid] == target: return mid (found).",
            "5. If array[mid] < target: set low = mid + 1 (search right half).",
            "6. If array[mid] > target: set high = mid - 1 (search left half).",
            "7. If loop ends without finding: return -1 (not found).",
            "8. Time complexity: O(log n). Space complexity: O(1) for iterative.",
        ],
        "measure electrical continuity": [
            "1. Turn off or disconnect the circuit before testing.",
            "2. Set the multimeter to the continuity or resistance mode.",
            "3. Touch the two probes together — the meter should beep or show ~0 ohms.",
            "4. Touch the probes to the two ends of the conductor being tested.",
            "5. A beep or low resistance indicates continuity (connected path).",
            "6. No beep or OL (over limit) indicates an open circuit (broken path).",
            "7. Never test continuity on live circuits.",
        ],
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._query_count: int = 0
        logger.info("TechnicalKnowledgeBase initialized")

    def query(self, question: str, domain: Optional[TechDomain] = None) -> KnowledgeResult:
        """Search for a technical answer using keyword matching, optionally filtered by domain."""
        with self._lock:
            self._query_count += 1

        question_lower = question.lower().strip()

        # Direct match
        if question_lower in self.KNOWLEDGE_BASE:
            entry = self.KNOWLEDGE_BASE[question_lower]
            if domain is None or entry["domain"] == domain:
                return KnowledgeResult(
                    query=question,
                    answer=entry["answer"],
                    confidence=0.95,
                    sources=["Technical Documentation", "IEEE Standards", "MIT OpenCourseWare"],
                    related_topics=self._get_related_topics(question_lower),
                )

        # Keyword-based search
        best_key: Optional[str] = None
        best_score: float = 0.0
        q_words = set(question_lower.split())

        for key, entry in self.KNOWLEDGE_BASE.items():
            if domain is not None and entry["domain"] != domain:
                continue
            key_words = set(key.split())
            overlap = len(q_words & key_words)
            score = overlap / max(len(q_words), len(key_words), 1)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= 0.25:
            entry = self.KNOWLEDGE_BASE[best_key]
            return KnowledgeResult(
                query=question,
                answer=entry["answer"],
                confidence=min(0.9, best_score * 1.3),
                sources=["Technical Documentation", "Stack Overflow", "Wikipedia"],
                related_topics=self._get_related_topics(best_key),
            )

        # Check HOW_TO_GUIDES
        for guide_key, steps in self.HOW_TO_GUIDES.items():
            if any(word in question_lower for word in guide_key.split()):
                return KnowledgeResult(
                    query=question,
                    answer="How-to guide: " + " | ".join(steps[:3]) + " ...",
                    confidence=0.7,
                    sources=["Technical Documentation"],
                    related_topics=[guide_key],
                )

        return KnowledgeResult(
            query=question,
            answer=f"No specific technical information found for '{question}'. Try rephrasing or consult official documentation.",
            confidence=0.1,
            sources=[],
            related_topics=[],
        )

    def get_how_to(self, task: str) -> List[str]:
        """Return step-by-step guide for a given technical task."""
        task_lower = task.lower().strip()
        # Direct match
        if task_lower in self.HOW_TO_GUIDES:
            return self.HOW_TO_GUIDES[task_lower]
        # Fuzzy match
        best_key: Optional[str] = None
        best_score: float = 0.0
        task_words = set(task_lower.split())
        for key in self.HOW_TO_GUIDES:
            key_words = set(key.split())
            overlap = len(task_words & key_words)
            score = overlap / max(len(task_words), len(key_words), 1)
            if score > best_score:
                best_score = score
                best_key = key
        if best_key and best_score >= 0.3:
            return self.HOW_TO_GUIDES[best_key]
        return [f"No how-to guide found for '{task}'. Please check official documentation."]

    def explain_concept(self, concept: str, complexity_level: int = 3) -> str:
        """
        Explain a concept at a specified complexity level.
        1 = simple (child), 3 = intermediate, 5 = expert.
        """
        concept_lower = concept.lower().strip()
        complexity_level = max(1, min(5, complexity_level))

        # Find the base answer
        base_answer = ""
        for key, entry in self.KNOWLEDGE_BASE.items():
            if concept_lower in key or any(word in key for word in concept_lower.split() if len(word) > 3):
                base_answer = entry["answer"]
                break

        if not base_answer:
            return f"Concept '{concept}' not found in technical knowledge base."

        if complexity_level == 1:
            # Simplified: take first sentence
            first_sentence = base_answer.split(".")[0] + "."
            return f"Simply put: {first_sentence}"
        elif complexity_level == 2:
            # First two sentences
            sentences = base_answer.split(".")
            return ". ".join(sentences[:2]).strip() + "."
        elif complexity_level == 3:
            return base_answer
        elif complexity_level == 4:
            return base_answer + " For deeper understanding, explore related algorithms, data structures, and real-world applications."
        else:  # level 5
            return (
                base_answer
                + " Advanced considerations include performance trade-offs, edge cases, theoretical foundations, "
                  "and current research directions in this domain."
            )

    def search(self, keywords: List[str]) -> List[KnowledgeResult]:
        """Search technical knowledge base using multiple keywords."""
        results: List[KnowledgeResult] = []
        kw_set = {kw.lower() for kw in keywords}

        for key, entry in self.KNOWLEDGE_BASE.items():
            key_words = set(key.split())
            answer_words = set(entry["answer"].lower().split())
            overlap = len(kw_set & (key_words | answer_words))
            if overlap > 0:
                score = overlap / len(kw_set)
                results.append(KnowledgeResult(
                    query=" ".join(keywords),
                    answer=entry["answer"],
                    confidence=min(0.95, score),
                    sources=["Technical Documentation"],
                    related_topics=self._get_related_topics(key),
                ))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:10]

    def _get_related_topics(self, topic_key: str) -> List[str]:
        """Return related topics for a knowledge base key."""
        relations: Dict[str, List[str]] = {
            "python": ["programming", "machine learning", "web development", "automation"],
            "algorithm": ["data structures", "complexity", "sorting", "searching"],
            "machine learning": ["neural networks", "deep learning", "data science", "statistics"],
            "ohms law": ["resistors", "circuits", "voltage", "current"],
            "transistor": ["semiconductors", "amplifiers", "digital logic", "MOSFETs"],
            "tcp ip": ["networking", "OSI model", "UDP", "HTTP", "DNS"],
            "calculus": ["derivatives", "integrals", "differential equations", "physics"],
            "encryption": ["cybersecurity", "TLS", "RSA", "AES", "hash functions"],
            "database": ["SQL", "NoSQL", "indexing", "normalization", "ACID"],
            "api": ["REST", "microservices", "web services", "JSON", "HTTP"],
        }
        for key, related in relations.items():
            if key in topic_key:
                return related
        return ["technical documentation", "programming", "engineering"]

    def get_stats(self) -> Dict:
        """Return operational statistics."""
        with self._lock:
            return {
                "query_count": self._query_count,
                "total_knowledge_entries": len(self.KNOWLEDGE_BASE),
                "total_how_to_guides": len(self.HOW_TO_GUIDES),
                "domains": [d.value for d in TechDomain],
            }


_technical_kb_instance: Optional[TechnicalKnowledgeBase] = None
_technical_kb_lock = threading.Lock()


def get_technical_knowledge_base() -> TechnicalKnowledgeBase:
    """Return the singleton TechnicalKnowledgeBase instance."""
    global _technical_kb_instance
    if _technical_kb_instance is None:
        with _technical_kb_lock:
            if _technical_kb_instance is None:
                _technical_kb_instance = TechnicalKnowledgeBase()
    return _technical_kb_instance
