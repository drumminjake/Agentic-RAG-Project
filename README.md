# Agentic RAG: Router-Retriever System with PDF and Web Search Tools

A multi-agent Retrieval-Augmented Generation (RAG) system built with **CrewAI**. A **Router Agent** classifies each user question, and a **Retriever Agent** answers it from the right source: a PDF knowledge base (NIST AI RMF 1.0), a live web search (Tavily), or the LLM's general knowledge. Every run is logged for reasoning traceability.

**Files**

| File | Purpose |
|---|---|
| `agentic_rag.ipynb` | The working notebook: setup, tools, agents, tasks, crew, logging, and demo |
| `nist_ai_rmf.pdf` | Domain document: NIST AI 100-1, *AI Risk Management Framework (AI RMF 1.0)*, January 2023 |
| `trace_log.txt` | Clean trace log: question, each agent's output, route, reason, answer, and timing |
| `crew_verbose_log.txt` | CrewAI's detailed execution log, including tool calls |
| `.env` | API keys (`OPENAI_API_KEY`, `TAVILY_API_KEY`), **not submitted** |

---

## 1. System Architecture

```
                 ┌──────────────────────┐
  User question ─►     Router Agent     │  no tools; decides the source
                 │   ("Query Router")   │  output: ROUTE + REASON
                 └──────────┬───────────┘
                            │  context hand-off (Task 1 output → Task 2)
                            ▼
                 ┌──────────────────────┐
                 │   Retriever Agent    │
                 │("Information Retriever")
                 └───┬──────────┬───────┬┘
          ROUTE: PDF │ ROUTE: WEB│       │ ROUTE: LLM
                     ▼          ▼       ▼
             PDFSearchTool   Tavily   LLM only
            (vector search)  (web)   (no retrieval)
                     └──────────┼───────┘
                                ▼
                   Grounded answer + Sources
                                │
                                ▼
              trace_log.txt / crew_verbose_log.txt / summary table
```

**Components**

- **LLM:** OpenAI `gpt-4o-mini` at `temperature=0`, shared by both agents.
- **PDF tool:** `crewai_tools.PDFSearchTool`. It splits the PDF into chunks, embeds them with OpenAI embeddings, stores them in a local vector database, and returns the passages closest in meaning to the query, with page numbers.
- **Web tool:** LangChain's `TavilySearchResults` (top 3 results), wrapped in a custom CrewAI `BaseTool` (`WebSearchTool`) that formats each result with its source URL.
- **Orchestration:** a CrewAI `Crew` running two `Task`s with `Process.sequential`.

---

## 2. Agent Logic and Responsibilities

### Router Agent: "Query Router"
- **Responsibility:** analyze the question and choose exactly one source. It never answers the question itself.
- **Tools:** none, so its only job is one clear decision.
- **Decision rules (in its backstory):**
  - **PDF:** questions about the content of NIST AI RMF 1.0, such as AI risk concepts, trustworthy-AI characteristics, and the Govern/Map/Measure/Manage functions.
  - **WEB:** current or time-sensitive questions (news, updates after January 2023) or topics the PDF does not cover.
  - **LLM:** general knowledge that needs no source, such as common definitions.
- **Output contract:** `ROUTE: <PDF|WEB|LLM>` and `REASON: <one sentence>`.

### Retriever Agent: "Information Retriever"
- **Responsibility:** follow the router's decision, retrieve from that source, and write a concise answer based only on what was retrieved.
- **Tools:** `PDFSearchTool` and `WebSearchTool`. It uses only the tool named by the route, and no tool for `LLM`.
- **Output contract:** a 1–3 paragraph answer ending in a `Sources:` line (PDF page numbers, URLs, or "General knowledge (no retrieval)").

Both agents use `allow_delegation=False`, so control flows in one direction only: Router to Retriever.

---

## 3. Coordination Flow

1. The user calls `await ask("<question>")`.
2. `ask()` builds the crew and starts it with `crew.kickoff_async(inputs={"question": ...})`. The `{question}` placeholder is filled into both task descriptions.
3. **Task 1 (`route_task`)**: the Router Agent returns `ROUTE` and `REASON`.
4. The output of Task 1 is passed to Task 2 through `context=[route_task]`. This is the explicit input-output hand-off between the agents.
5. **Task 2 (`retrieve_task`)**: the Retriever Agent reads the route, calls the matching tool (or none), and writes the grounded answer.
6. Logging runs throughout:
   - `task_callback` writes each agent's output to `trace_log.txt` as soon as its task finishes.
   - `ask()` logs the question, the parsed route and reason, the final answer, and the elapsed time.
   - `output_log_file` stores CrewAI's full verbose trace, including tool calls, in `crew_verbose_log.txt`.
   - Each run is also added to `trace_records`, which the notebook displays as a summary table and per-question trace view.

### Demo results

| # | Question | Route chosen | Expected |
|---|---|---|---|
| 1 | What does the GOVERN function cover in the NIST AI RMF? | PDF | PDF ✅ |
| 2 | What updates or new profiles has NIST released for the AI RMF in 2025 or 2026? | WEB | WEB ✅ |
| 3 | In simple terms, what is a large language model? | LLM | LLM ✅ |

An additional test question ("What are the characteristics of trustworthy AI according to the NIST AI RMF?") was routed to PDF and answered with page citations.

---

## 4. Why CrewAI

- **Role-based design fits the problem.** The assignment is defined in terms of roles (router, retriever). CrewAI's `Agent(role, goal, backstory)` maps directly onto that, so each agent's responsibility is explicit and readable.
- **Built-in task hand-off.** `Task(context=[...])` passes one agent's output to the next without custom plumbing, which keeps the input-output flow clear.
- **RAG tools included.** `crewai_tools.PDFSearchTool` handles chunking, embedding, and vector search in one line.
- **Interoperability.** LangChain tools such as `TavilySearchResults` can be used through a small `BaseTool` wrapper.
- **Traceability.** `verbose=True`, `task_callback`, and `output_log_file` provide reasoning logs with little extra code.

**Alternatives considered:** LangGraph offers finer control over branching (an explicit graph edge per route), but takes more code to set up. Plain LangChain lacks first-class multi-agent roles. For a two-agent sequential workflow, CrewAI gives the clearest design with the least code.

---

## 5. Challenges Faced and Trade-offs

**Challenges**

- **Jupyter event loop conflict.** The installed CrewAI version raises `RuntimeError: Agent execution was invoked synchronously from within a running event loop` when `crew.kickoff()` is called in Jupyter, because Jupyter already runs an asyncio event loop. *Fix:* use `await crew.kickoff_async(...)` and make `ask()` an `async` function.
- **Framework mismatch for the web tool.** `TavilySearchResults` is a LangChain tool, not a CrewAI tool. *Fix:* a custom `BaseTool` subclass wraps it and formats each result with its URL so answers can cite sources.
- **Deprecation warning.** LangChain marks `TavilySearchResults` as deprecated in favor of a newer package. It is kept because the assignment specifies it, and it still works.
- **Environment setup.** The virtual environment was first created in a different folder from the notebook. The fix was to rebuild it inside the project folder and select it as the Jupyter kernel.
- **Grounded is not always complete.** In one test, the agent listed six trustworthy-AI characteristics when NIST AI RMF 1.0 defines seven. It merged "Safe" into "Secure and Resilient." Retrieval was correct, but the smaller model summarized imperfectly. Answers should still be checked against the source.

**Trade-offs**

| Decision | Benefit | Cost |
|---|---|---|
| `gpt-4o-mini`, `temperature=0` | Cheap, fast, consistent routing | Less thorough summaries than a larger model |
| Routing by LLM prompt (not keyword rules) | Handles varied phrasing; gives a readable reason | Depends on backstory wording; can occasionally misroute |
| Retriever holds both tools and follows the route | Simple, single crew; shows agent collaboration | Relies on the agent obeying instructions rather than a hard code-level restriction |
| One small, focused PDF (48 pages) | Fast indexing, low embedding cost, precise retrieval | Knowledge base is limited to one document from January 2023 |
| Top 3 Tavily results | Short context, lower cost | May miss information found in lower-ranked results |
| Optional LLM-only route | Fast answers for general questions | Those answers have no external source to cite |

**Possible improvements:** enforce routing in code by giving the Retriever only the chosen tool; add a fallback (for example, PDF → WEB when the PDF has no answer); use a larger model for summarization; and add an evaluation set of labeled questions to measure routing accuracy.

---

## 6. How to Run

1. Python 3.10–3.13. Create and activate a virtual environment in the project folder:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install crewai crewai-tools langchain-community tavily-python python-dotenv ipykernel
   ```
2. Create `.env` in the project folder:
   ```
   OPENAI_API_KEY=your-openai-key
   TAVILY_API_KEY=your-tavily-key
   ```
3. Place `nist_ai_rmf.pdf` in the project folder (source: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf).
4. Open `agentic_rag.ipynb` in VSCode, select the `.venv` kernel, and run all cells in order.
