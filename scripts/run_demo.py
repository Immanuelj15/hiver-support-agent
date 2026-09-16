"""
scripts/run_demo.py

Interactive and batch CLI demonstration for Hiver AI Customer Support Agent.
Supports single queries, batch query files, and interactive REPL mode across both
offline Ollama and Cloud LLM providers.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.support_agent import SupportAgent
from src.llm.factory import get_llm_provider

def format_terminal_output(res: dict) -> None:
    """Print readable audit trail safe for all terminal encodings."""
    print("\n" + "=" * 70)
    print("AI CUSTOMER SUPPORT AGENT EXECUTION")
    print("=" * 70)
    print(f"Customer Inquiry:      {res['customer_msg']}")
    print(f"Predicted Intent:      {res['intent']} (Confidence: {res['intent_confidence']:.2f})")
    print(f"Retrieval Similarity:  {res['retrieval_similarity']:.4f}")
    print(f"Pipeline Action:       {res['action']} (Risk: {res['risk_level']})")
    print(f"Escalation Flag:       {'YES (Escalate to human)' if res['should_escalate'] else 'NO (Auto-handled)'}")
    if res['should_escalate']:
        print(f"Escalation Reason:     {res['escalation_reason']}")
        if res.get('triggered_rule'):
            print(f"Triggered Rule:        {res['triggered_rule']}")
    print(f"Engine:                {res['provider']} ({res['model']})")
    print("-" * 70)
    print(f"Draft Response:\n{res['reply']}")
    print("-" * 70)
    if res.get("retrieved_evidence"):
        print(f"Top Retrieved Grounding Case:")
        ev = res["retrieved_evidence"][0]
        print(f"  [Ref: {ev.get('conversation_id')}] (Sim: {ev.get('similarity'):.4f})")
        clean_cust = ev.get('customer_msg', '').replace('\n', ' ')[:75]
        clean_ans = ev.get('agent_reply', '').replace('\n', ' ')[:75]
        print(f"  Customer: {clean_cust}...")
        print(f"  Agent:    {clean_ans}...")
    print("=" * 70 + "\n")

def run_interactive(agent: SupportAgent) -> None:
    """Run interactive terminal REPL."""
    print("\n" + "=" * 70)
    print("Hiver AI Support Agent - Interactive Session")
    print(f"Provider: {agent.llm.provider_type} | Model: {agent.llm.model_name}")
    print("Type your customer message below (or 'exit' / 'quit' to end):")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("Customer > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive session. Goodbye!")
                break

            result = agent.process(user_input)
            format_terminal_output(result)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"Error processing request: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Hiver AI Customer Support Agent.")
    parser.add_argument("--message", type=str, help="Single customer message string")
    parser.add_argument("--file", type=str, help="Path to text file containing one query per line")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session")
    parser.add_argument("--provider", type=str, choices=["ollama", "cloud"], help="Override LLM provider")
    parser.add_argument("--model", type=str, help="Override LLM model name")
    parser.add_argument("--json", action="store_true", help="Output full JSON instead of formatted text")
    args = parser.parse_args()

    # Instantiate provider and agent
    llm = get_llm_provider(provider=args.provider, model=args.model)
    agent = SupportAgent(llm_provider=llm)

    if args.interactive:
        run_interactive(agent)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"Processing {len(queries)} queries from {args.file}...\n")
        for q in queries:
            res = agent.process(q)
            if args.json:
                print(json.dumps(res, ensure_ascii=False))
            else:
                format_terminal_output(res)
    elif args.message:
        res = agent.process(args.message)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            format_terminal_output(res)
    else:
        # Default behavior: run on a sample query
        default_query = "Where is my parcel? It was supposed to be here yesterday."
        print(f"No query provided. Running default sample query: '{default_query}'")
        res = agent.process(default_query)
        format_terminal_output(res)

if __name__ == "__main__":
    main()
