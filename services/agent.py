"""
PropIQ Agent
------------
The agentic layer. Given a natural-language real-estate question, the
agent (Claude, via tool-use) decides which tools to call - the
TensorFlow-backed valuation service, the mortgage calculator, the
comparables lookup, the neighborhood stats - potentially calling
several in sequence, then synthesizes a final answer grounded in
their outputs.

This is genuinely agentic (not a single prompt->answer call): the
model plans, calls tools, observes results, and can call more tools
before answering - a real tool-use loop, not a scripted pipeline.

Requires ANTHROPIC_API_KEY to be set in the environment to run live.
"""
import json
import os

import httpx
from anthropic import Anthropic

from services import tools

VALUATION_SERVICE_URL = os.environ.get("VALUATION_SERVICE_URL", "http://localhost:8001")

SYSTEM_PROMPT = """You are PropIQ, a real estate investment advisor assistant for the Dubai market.

You have tools to: get an ML-based price estimate for a property, look up comparable listings,
estimate a mortgage, and pull neighborhood statistics. Use as many tools as needed - typically a
valuation, then comparables and/or neighborhood stats to sanity-check it, and a mortgage estimate
if the user mentions financing or affordability - before giving your final answer.

Always ground your recommendation in the actual tool outputs (cite the numbers). Be direct about
uncertainty - the price model gives an estimate with a range, not a guarantee. Keep the final
answer concise and structured: verdict, key numbers, and 2-3 sentences of reasoning."""

TOOL_DEFINITIONS = [
    {
        "name": "predict_price",
        "description": "Get a TensorFlow ML-based price estimate (and rent/yield estimate) for a property based on its features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "area": {"type": "string"},
                "property_type": {"type": "string", "enum": ["Apartment", "Townhouse", "Villa"]},
                "bedrooms": {"type": "integer"},
                "size_sqft": {"type": "number"},
                "building_age_years": {"type": "integer", "default": 0},
                "near_metro": {"type": "boolean", "default": False},
                "has_pool": {"type": "boolean", "default": False},
            },
            "required": ["area", "property_type", "bedrooms", "size_sqft"],
        },
    },
    {
        "name": "get_comparable_listings",
        "description": "Look up comparable listings in the same area/bedroom count to sanity-check a valuation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "area": {"type": "string"},
                "bedrooms": {"type": "integer"},
                "property_type": {"type": "string"},
            },
            "required": ["area", "bedrooms"],
        },
    },
    {
        "name": "get_neighborhood_stats",
        "description": "Get aggregate market stats (median price, price/sqft, typical yield) for a neighborhood.",
        "input_schema": {
            "type": "object",
            "properties": {"area": {"type": "string"}},
            "required": ["area"],
        },
    },
    {
        "name": "estimate_mortgage",
        "description": "Estimate monthly mortgage payment and total interest for a given price and financing terms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "price_aed": {"type": "number"},
                "down_payment_pct": {"type": "number", "default": 20},
                "annual_rate_pct": {"type": "number", "default": 4.0},
                "term_years": {"type": "integer", "default": 25},
            },
            "required": ["price_aed"],
        },
    },
]


def _call_valuation_service(args: dict) -> dict:
    payload = {
        "area": args["area"],
        "property_type": args["property_type"],
        "bedrooms": args["bedrooms"],
        "size_sqft": args["size_sqft"],
        "building_age_years": args.get("building_age_years", 0),
        "near_metro": args.get("near_metro", False),
        "has_pool": args.get("has_pool", False),
    }
    resp = httpx.post(f"{VALUATION_SERVICE_URL}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _execute_tool(name: str, args: dict) -> dict:
    if name == "predict_price":
        return _call_valuation_service(args)
    if name == "get_comparable_listings":
        return {"listings": tools.get_comparable_listings(**args)}
    if name == "get_neighborhood_stats":
        return tools.get_neighborhood_stats(**args)
    if name == "estimate_mortgage":
        return tools.estimate_mortgage(**args)
    raise ValueError(f"Unknown tool: {name}")


def run_agent(user_query: str, max_turns: int = 6, model: str = "claude-sonnet-4-6") -> str:
    """Runs the full agent loop: plan -> call tools -> observe -> repeat -> answer."""
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    messages = [{"role": "user", "content": user_query}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = _execute_tool(block.name, block.input)
                content = json.dumps(result)
            except Exception as exc:  # tool errors are fed back to the model, not raised
                content = json.dumps({"error": str(exc)})

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content}
            )

        messages.append({"role": "user", "content": tool_results})

    return "Reached max tool-use turns without a final answer."


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or (
        "Should I buy a 2 bedroom apartment in JVC around 1100 sqft for AED 950,000? "
        "I'd put 20% down. What's a fair rent estimate too?"
    )
    print(run_agent(query))
