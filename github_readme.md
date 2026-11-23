# Python KI-Agent mit Tools, Live-API, RAG & Langzeitgedächtnis

Dieses Repository enthält eine minimalistische, voll funktionsfähige Implementierung eines KI-Agenten in Python mit dem offiziellen **Google GenAI SDK** – ganz ohne schweres Framework.

Der Agent fungiert als Shop-Assistent und demonstriert alle Kernkonzepte moderner KI-Agenten:
- Eigene lokale Tools (Lagerbestand & Preise)
- Echte Live-API-Aufrufe (Wechselkurse über die Frankfurter-API)
- Agentic RAG (Durchsuchen eigener PDF-Dokumente wie AGB oder Handbücher)
- Dauerhaftes Langzeitgedächtnis über Neustarts hinweg (ChromaDB)

---

## 1. Was ist ein KI-Agent? (Die 4 Kernbausteine)

Im Kern besteht ein KI-Agent aus einem großen Sprachmodell (LLM) als „Gehirn“, das in einer Schleife selbstständig plant, externe Werkzeuge bedient und Ergebnisse überprüft, bis eine Aufgabe gelöst ist:

1. **Gehirn (LLM):** Übernimmt Sprachverständnis, Planung und Entscheidungsfindung (hier: `gemini-2.5-flash`).
2. **Werkzeuge (Tools):** Schnittstellen (Funktionen/APIs), mit denen der Agent aktiv handeln oder Daten abrufen kann.
3. **Gedächtnis (Memory):** Kurzzeitgedächtnis für den aktuellen Aufgabenverlauf sowie Langzeitgedächtnis und Fachwissen über eine Vektordatenbank (RAG).
4. **Steuerungsschleife (Agent Loop):** Die Logik (oft nach dem ReAct-Prinzip: *Reason -> Act -> Observe*), die den Agenten Schritt für Schritt arbeiten lässt, bis das Ziel erreicht ist.

### Übersicht der Ansätze
- **No-Code / Low-Code** (n8n, Make, Zapier, Flowise): Ideal für schnelle Prototypen und Büro-Automatisierung.
- **Code-Frameworks** (LangGraph, CrewAI, Google ADK, OpenAI Agents SDK): Ideal für komplexe Multi-Agenten-Systeme.
- **Eigenbau ohne Framework (dieses Projekt):** Maximale Transparenz, minimale Abhängigkeiten und tiefes Verständnis der Funktionsweise.

---

## 2. Vorbereitung & Installation

### Voraussetzungen
- Python 3.10 oder neuer
- Ein kostenloser Gemini API-Key aus dem Google AI Studio

### Abhängigkeiten installieren
```bash
pip install google-genai requests chromadb pypdf
```

### API-Key als Umgebungsvariable setzen

**Linux / macOS:**
```bash
export GEMINI_API_KEY="dein-api-key"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="dein-api-key"
```

---

## 3. Vollständiger Python-Code (`agent.py`)

Speichere den folgenden Code als `agent.py` ab. Er vereint alle Tools (Lagerbestand, Live-Währungsrechner, PDF-Suche/RAG und Langzeitgedächtnis) in einer gemeinsamen Agenten-Schleife.

```python
import uuid
import requests
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Initialisierung: LLM-Client & lokale Vektordatenbank
# ---------------------------------------------------------
# Liest GEMINI_API_KEY automatisch aus den Umgebungsvariablen
client = genai.Client()

# Dauerhafte lokale Vektordatenbank (speichert auf der Festplatte)
db_client = chromadb.PersistentClient(path="./agent_db")
docs_collection = db_client.get_or_create_collection(name="shop_docs")
memory_collection = db_client.get_or_create_collection(name="user_memory")


# ---------------------------------------------------------
# 2. Hilfsfunktion: PDFs einmalig für RAG indexieren
# ---------------------------------------------------------
def index_pdf(file_path: str):
    """Liest eine PDF-Datei seitenweise ein und speichert sie in ChromaDB."""
    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # upsert verhindert Duplikate bei erneutem Ausführen
            docs_collection.upsert(
                documents=[text],
                ids=[f"{file_path}_seite_{i + 1}"],
                metadatas=[{"source": file_path, "page": i + 1}],
            )
    print(f"PDF '{file_path}' erfolgreich indexiert.")


# ---------------------------------------------------------
# 3. Definition der Agenten-Tools
#    (Docstrings & Typ-Hinweise sind die Anleitung fürs LLM!)
# ---------------------------------------------------------

# Tool 1: Lokales Lager
def check_inventory(product: str) -> dict:
    """Prüft den aktuellen Lagerbestand und Preis eines Produkts in Euro."""
    inventory = {
        "laptop": {"stock": 4, "price_eur": 999},
        "maus": {"stock": 23, "price_eur": 29},
        "tastatur": {"stock": 0, "price_eur": 79},
    }
    return inventory.get(
        product.lower(),
        {"stock": 0, "price_eur": None, "info": "Produkt nicht gefunden"},
    )


# Tool 2: Echtes Live-API-Tool (EZB-Wechselkurse ohne API-Key)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Rechnet einen Geldbetrag mit Live-Wechselkursen um (z. B. EUR in USD, CHF oder GBP)."""
    url = "https://api.frankfurter.app/latest"
    params = {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "converted_amount": data["rates"][to_currency.upper()],
            "currency": to_currency.upper(),
            "date": data["date"],
        }
    except requests.RequestException as e:
        # Fehler als Text zurückgeben, damit der Agent sinnvoll reagieren kann
        return {"error": f"API-Abfrage fehlgeschlagen: {e}"}


# Tool 3: RAG-Tool für eigene PDF-Dokumente
def search_documents(query: str) -> list[str]:
    """Durchsucht interne PDF-Dokumente (z. B. AGB, Rückgaberichtlinien, Handbücher)."""
    results = docs_collection.query(query_texts=[query], n_results=2)
    return results["documents"][0] if results["documents"] else ["Keine Treffer."]


# Tool 4 & 5: Langzeitgedächtnis (Speichern & Abrufen)
def save_memory(fact: str) -> str:
    """Speichert wichtige Infos über den Nutzer dauerhaft (z. B. Name, Vorlieben, Währung)."""
    memory_collection.add(documents=[fact], ids=[str(uuid.uuid4())])
    return f"Gemerkt: {fact}"


def recall_memory(query: str) -> list[str]:
    """Ruft gespeicherte Infos über den Nutzer aus dem Langzeitgedächtnis ab."""
    results = memory_collection.query(query_texts=[query], n_results=3)
    return results["documents"][0] if results["documents"] else ["Keine Erinnerungen."]


# ---------------------------------------------------------
# 4. Tool-Registrierung & Agenten-Konfiguration
# ---------------------------------------------------------
tools_map = {
    "check_inventory": check_inventory,
    "convert_currency": convert_currency,
    "search_documents": search_documents,
    "save_memory": save_memory,
    "recall_memory": recall_memory,
}

config = types.GenerateContentConfig(
    system_instruction=(
        "Du bist ein hilfreicher Shop-Assistent. "
        "Nutze deine Tools für Lagerbestände, Preise und Live-Währungsumrechnungen. "
        "Nutze 'search_documents' für Fragen zu AGB, Garantie oder Rückgabe. "
        "Nutze 'save_memory', wenn der Nutzer persönliche Vorlieben nennt (z. B. bevorzugte Währung), "
        "und 'recall_memory', um dich an frühere Angaben zu erinnern. "
        "Wenn eine API ausfällt, informiere den Kunden ehrlich darüber."
    ),
    tools=list(tools_map.values()),
    # Automatische Aufrufe ausschalten, um die Schleife selbst zu kontrollieren
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# ---------------------------------------------------------
# 5. Die Agenten-Schleife (Agent Loop)
# ---------------------------------------------------------
def run_agent(user_prompt: str, max_steps: int = 5):
    # Kurzzeitgedächtnis (aktueller Gesprächsverlauf)
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    ]

    for step in range(max_steps):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        # Antwort des Modells im Verlauf speichern
        contents.append(response.candidates[0].content)

        # Wenn kein Tool mehr benötigt wird -> Finale Antwort ausgeben
        if not response.function_calls:
            print(f"\nAgent: {response.text}")
            return

        # Alle vom Modell angeforderten Tools ausführen
        tool_parts = []
        for call in response.function_calls:
            print(f"[Schritt {step + 1}] Tool-Aufruf: {call.name}({call.args})")
            func = tools_map[call.name]
            result = func(**call.args)

            tool_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        # Tool-Ergebnisse zurück an das LLM senden
        contents.append(types.Content(role="user", parts=tool_parts))

    print("Abbruch: Maximale Anzahl an Schritten erreicht.")


# ---------------------------------------------------------
# 6. Ausführen & Testen
# ---------------------------------------------------------
if __name__ == "__main__":
    # Optional: Eigene PDF-Datei vor dem ersten Lauf indexieren
    # index_pdf("agb.pdf")

    # Beispiel 1: Tool-Chaining (Lager prüfen -> Live-Umrechnung in USD)
    run_agent("Habt ihr noch einen Laptop auf Lager? Und was kostet er aktuell in US-Dollar?")

    # Beispiel 2: Langzeitgedächtnis testen
    # run_agent("Merk dir bitte, dass ich Preise immer gerne in Schweizer Franken (CHF) sehe.")
```

### Starten
```bash
python agent.py
```

---

## 4. Wichtige Architektur-Konzepte & Best Practices

- **Docstrings & Typ-Hinweise sind entscheidend:** Das LLM sieht nicht den internen Python-Code, sondern nur Funktionsnamen, Parameter-Typen und Docstrings. Präzise Beschreibungen sorgen für zuverlässige Tool-Aufrufe.
- **Automatisches Tool-Chaining:** Bei komplexen Fragen verkettet der Agent mehrere Schritte selbstständig (z. B. erst `check_inventory("laptop")` -> dann mit dem Ergebnis `convert_currency(999, "EUR", "USD")`).
- **Agentic RAG & Memory:** Dokumente und Erinnerungen werden nicht blind in jeden Prompt geladen, sondern nur bei Bedarf aktiv über Tools abgerufen. Das spart Tokens und reduziert Halluzinationen.
- **Leitplanken (Guardrails):**
  - `max_steps` begrenzt die Schleifendurchläufe und verhindert Endlosschleifen sowie unnötige API-Kosten.
  - Fehlerbehandlung (`try/except` + `timeout`) bei externen APIs verhindert Abstürze und erlaubt dem Agenten eine saubere Antwort.
  - Bei kritischen Aktionen (z. B. E-Mails versenden oder Bestellungen auslösen) sollte zusätzlich eine **Human-in-the-Loop**-Bestätigung eingebaut werden.
- **Tipp für große PDFs:** In diesem Beispiel wird jede PDF-Seite als ein Block gespeichert. Für sehr umfangreiche Dokumente empfiehlt sich später **Chunking** (Zerlegen in Textblöcke von ca. 500–1.000 Zeichen mit leichtem Überlapp).