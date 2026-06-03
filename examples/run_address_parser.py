"""Address parser API powered by specllm + Bedrock Claude."""
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

print("🚀 Address Parser API running on http://127.0.0.1:8080")
print("   POST /v1/parse-address  {\"address\": \"...\"}")
print("   Ctrl+C to stop\n")
app.serve(port=8080)
