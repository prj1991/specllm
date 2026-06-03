#!/usr/bin/env python
"""Interactive specllm demo — parse addresses via LLM.

Usage:
    python examples/demo.py
    python examples/demo.py "742 Evergreen Terrace, Springfield, IL 62704"
"""
import json
import sys

sys.path.insert(0, ".")

import boto3
from specllm import SpecLLM
from specllm.llm.providers import LLMProvider


class BedrockProvider(LLMProvider):
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")

    def call(self, prompt, system_prompt=None):
        response = self.client.invoke_model(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "system": "Respond ONLY with valid JSON. No markdown, no explanation.",
                "messages": [{"role": "user", "content": prompt}],
            }),
            contentType="application/json",
        )
        text = json.loads(response["body"].read())["content"][0]["text"].strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        return json.loads(text)


app = SpecLLM.from_openapi("./examples/address_parser_spec.json", provider=BedrockProvider())
client = app.test_client()


def query(address):
    print(f"\n📍 Input: {address}")
    result = client.post("/v1/parse-address", json_body={"address": address})
    if "error" in result:
        print(f"❌ Error: {result['error']['message']}")
    else:
        print(f"🏙️  City:    {result['city']}")
        print(f"📮 Pincode: {result['pincode']}")
    return result


if __name__ == "__main__":
    # If args provided, run those and exit
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
        sys.exit(0)

    # Interactive mode
    print("=" * 50)
    print("specllm Address Parser Demo")
    print("=" * 50)
    print("Type an address and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            address = input("Address> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not address or address.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        query(address)
        print()
