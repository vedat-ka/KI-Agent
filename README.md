# 🤖 Eigener KI-Agent in Python (Gemini API + Live-API + RAG + Memory)

Ein minimalistisches, voll funktionsfähiges Beispiel für einen KI-Agenten in Python – ganz ohne schweres Framework, um die Funktionsweise der **Agenten-Schleife (Agent Loop)** unter der Haube transparent zu verstehen.

Der Agent fungiert als Shop-Assistent und kombiniert:
- **Lokale Tools:** Lagerbestand & Preise prüfen (`check_inventory`)
- **Live-API-Tool:** Echtzeit-Währungsumrechnung über die Frankfurter-API (`convert_currency`)
- **Agentic RAG:** Eigene PDF-Dokumente (z. B. AGB, Handbücher) lokal durchsuchen (`search_documents`)
- **Langzeitgedächtnis:** Nutzervorlieben dauerhaft speichern und abrufen (`save_memory`, `recall_memory`)

---

## 🧠 Die 4 Kernbausteine eines KI-Agenten

| Baustein | Aufgabe | Umsetzung in diesem Projekt |
| :--- | :--- | :--- |
| **Gehirn (LLM)** | Sprachverständnis, Planung & Entscheidungen | `gemini-2.5-flash` über das Google GenAI SDK |
| **Werkzeuge (Tools)** | Handeln über externe APIs & Funktionen | Python-Funktionen mit Docstrings & Typ-Hinweisen |
| **Gedächtnis (Memory)** | Kurzzeitverlauf & dauerhaftes Wissen (RAG) | Chat-Liste (`contents`) + lokale **ChromaDB** |
| **Steuerungsschleife** | Schrittweises Arbeiten (*Reason → Act → Observe*) | Eigene `for`-Schleife mit `max_steps`-Limit |

> 💡 **Interaktive Visualisierung:** Öffne die Datei [`ki_agenten_zyklus_agent_loop.html`](./ki_agenten_zyklus_agent_loop.html) im Browser (oder über GitHub Pages), um den gesamten **KI-Agenten-Zyklus (Agent Loop)** Schritt für Schritt interaktiv durchzugehen.

---

## 🛠️ Installation & Setup

### 1. Repository klonen & Pakete installieren

```bash
git clone [https://github.com/DEIN-NAME/DEIN-REPO.git](https://github.com/DEIN-NAME/DEIN-REPO.git)
cd DEIN-REPO
pip install -r requirements.txt
```

Falls du die Pakete manuell installieren möchtest:

```bash
pip install google-genai requests chromadb pypdf
```

### 2. API-Key hinterlegen

Erstelle einen kostenlosen API-Key in [Google AI Studio](https://aistudio.google.com/) und setze ihn als Umgebungsvariable:

**macOS / Linux:**
```bash
export GEMINI_API_KEY="dein-api-key"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="dein-api-key"
```

---

## 💻 Der komplette KI-Agent (`agent.py`)

```python
import uuid
import requests
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

# 1. Gemini-Client & lokale Vektordatenbank initialisieren
client = genai.Client()
db_client = chromadb.PersistentClient(path="./agent_db")
docs_collection = db_client.get_or_create_collection(name="shop_docs")
memory_collection = db_client.get_or_create_collection(name="user_memory")


# Optional: Einmalig eine PDF-Datei für RAG indexieren
def index_pdf(file_path: str):
    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs_collection.upsert(
                documents=[text],
                ids=[f"{file_path}_seite_{i + 1}"],
                metadatas=[{"source": file_path, "page": i + 1}],
            )


# --- TOOLS ---

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


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Rechnet einen Geldbetrag mit Live-Wechselkursen um (z. B. EUR in USD, CHF oder GBP)."""
    url = "[https://api.frankfurter.app/latest](https://api.frankfurter.app/latest)"
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
        return {"error": f"API-Abfrage fehlgeschlagen: {e}"}


def search_documents(query: str) -> list[str]:
    """Durchsucht interne PDF-Dokumente (z. B. AGB, Rückgaberichtlinien, Handbücher)."""
    results = docs_collection.query(query_texts=[query], n_results=2)
    return results["documents"][0] if results["documents"] else ["Keine Treffer."]


def save_memory(fact: str) -> str:
    """Speichert wichtige Infos über den Nutzer dauerhaft (z. B. Name, Vorlieben, Währung)."""
    memory_collection.add(documents=[fact], ids=[str(uuid.uuid4())])
    return f"Gemerkt: {fact}"


def recall_memory(query: str) -> list[str]:
    """Ruft gespeicherte Infos über den Nutzer aus dem Langzeitgedächtnis ab."""
    results = memory_collection.query(query_texts=[query], n_results=3)
    return results["documents"][0] if results["documents"] else ["Keine Erinnerungen."]


# --- KONFIGURATION ---

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
        "Nutze 'save_memory', wenn der Nutzer persönliche Vorlieben nennt, "
        "und 'recall_memory', um dich an frühere Angaben zu erinnern. "
        "Wenn eine API ausfällt, informiere den Kunden ehrlich darüber."
    ),
    tools=list(tools_map.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# --- AGENTEN-SCHLEIFE (AGENT LOOP) ---

def run_agent(user_prompt: str, max_steps: int = 5):
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    ]

    for step in range(max_steps):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
        contents.append(response.candidates[0].content)

        # Kein weiterer Tool-Aufruf nötig -> Finale Antwort ausgeben
        if not response.function_calls:
            print(f"\nAgent: {response.text}")
            return

        # Alle angeforderten Tools ausführen
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

        contents.append(types.Content(role="user", parts=tool_parts))

    print("Abbruch: Maximale Anzahl an Schritten erreicht.")


if __name__ == "__main__":
    # Optional: index_pdf("agb.pdf")
    run_agent("Habt ihr noch einen Laptop auf Lager? Und was kostet er aktuell in US-Dollar?")
```

---

## 🔒 Best Practices & Sicherheit

- **Leitplanken (`max_steps`):** Begrenze immer die maximale Anzahl an Schleifendurchläufen, um Endlosschleifen und unnötige API-Kosten zu vermeiden.
- **Saubere Docstrings:** Das LLM sieht nicht den Funktionscode, sondern nur Funktionsname, Typ-Hinweise und Docstring. Je präziser diese sind, desto zuverlässiger funktioniert der Agent.
- **Fehlerbehandlung:** Fange Netzwerk- und API-Fehler (`try/except`) ab und gib sie als Text an das Modell zurück, statt das Programm abstürzen zu lassen.
- **Keine Secrets committen:** Achte darauf, dass `.env` und `agent_db/` in deiner `.gitignore` stehen.
