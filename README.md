# PropIQ — Agentic AI Property Valuation & Advisory Assistant


https://propiq-valuation.fly.dev 

An agentic AI assistant for the Dubai real estate market. Ask it a natural-language
question about a property, and it plans which tools to call - a TensorFlow-trained
price model, comparable listings, neighborhood stats, a mortgage calculator - and
synthesizes a grounded, structured recommendation.

## Why this exists

Built as a natural extension of production real estate AI work (a WhatsApp lead-qualification
assistant), this project adds a formal ML + agentic layer: instead of scripted conversation
flows, an LLM (Claude) decides which tools to call and in what order, based on the actual
question asked.

## Architecture

```
User query ("Should I buy a 2BR in JVC for 950k?")
        │
        ▼
  services/agent.py  ── Claude tool-use loop (the agentic layer)
        │
        ├── predict_price ─────────► services/valuation_service.py (FastAPI)
        │                                    │
        │                                    ▼
        │                            ml/train_model.py (TensorFlow regression model)
        │
        ├── get_comparable_listings ─► services/tools.py
        ├── get_neighborhood_stats ──► services/tools.py
        └── estimate_mortgage ───────► services/tools.py
        │
        ▼
  Final synthesized answer (verdict + numbers + reasoning)
```

**Why this counts as "agentic," not just an LLM call:** the model plans multi-step
tool sequences on its own (e.g. valuation -> comparables -> mortgage), observes each
tool's output, and can call further tools before answering - a real tool-use loop
(see `run_agent()` in `services/agent.py`), not a fixed pipeline.

## Stack

| Requirement | Where it lives |
|---|---|
| TensorFlow | `ml/train_model.py` - a Keras regression model predicting price from property features |
| LLM / Agentic AI | `services/agent.py` - Claude's tool-use API, no framework (LangChain etc.) needed |
| CI/CD pipeline | `.github/workflows/ci-cd.yml` - test -> train -> **model quality gate** -> Docker build -> push |
| Python | FastAPI (`services/valuation_service.py`), plain Python tools (`services/tools.py`) |

## Running it locally

```bash
pip install -r requirements.txt

# 1. Generate the (synthetic, stand-in) dataset and train the model
python ml/generate_data.py
python ml/train_model.py

# 2. Run the tests
python -m pytest tests/ -v

# 3. Start the valuation microservice
uvicorn services.valuation_service:app --reload --port 8001

# 4. In another terminal, run the agent (needs ANTHROPIC_API_KEY set)
export ANTHROPIC_API_KEY=sk-...
python services/agent.py "Should I buy a 2 bedroom apartment in JVC for AED 950,000?"
```

## CI/CD pipeline (`.github/workflows/ci-cd.yml`)

1. **test** - installs deps, generates data, trains the model, runs the full pytest suite.
2. **model-quality-gate** - retrains and checks the held-out MAPE against a threshold
   (`ml/check_model_quality.py`) - a bad retrain never proceeds to build/deploy.
3. **build-and-push** - (main branch only) builds the Docker image and pushes it to
   GitHub Container Registry, tagged with the commit SHA. The final deploy step is a
   placeholder - point it at your actual target (Cloud Run, ECS, a VM pull+restart, etc.).

## Notes on the data

`ml/generate_data.py` generates a synthetic-but-realistic Dubai residential dataset
(price/sqft by area, property type premiums, age decay, amenity effects) so the whole
pipeline is runnable without an external data dependency. Swap it for a real source
(Dubai Land Department transaction exports, a Kaggle Dubai real estate dataset) by
replacing the CSV `ml/train_model.py` reads from - the training and serving code
doesn't change.

## Known limitations (worth stating honestly in an interview)

- The dataset is synthetic, so the model's accuracy numbers (~13-27% MAPE depending on
  run) reflect synthetic noise, not real market performance - the point is the working
  ML -> serving -> agent -> CI/CD pipeline, not a production-accurate valuation model.
- `get_comparable_listings` / `get_neighborhood_stats` read from a static CSV standing
  in for a real listings database - swap for a real data source in production.
- The agent's `estimate_mortgage` uses standard fixed-rate amortization; UAE mortgages
  can have different structures (e.g. Islamic financing) not modeled here.
  <img width="1575" height="157" alt="image" src="https://github.com/user-attachments/assets/233bb5d2-eaf8-45d6-9b38-6b49e6e0095b" />
  <img width="711" height="172" alt="image" src="https://github.com/user-attachments/assets/30a6c66e-bee2-46d6-b8c0-781f1705cc7b" />
  <img width="1317" height="135" alt="image" src="https://github.com/user-attachments/assets/46f0ed37-1916-41d5-a49b-f0dfaf2c736c" />



