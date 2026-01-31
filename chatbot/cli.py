# terminal chatbot - talks to the API, prints back

import argparse
import sys

import httpx

import config


def chat_loop(api_base_url: str, timeout: float = 60.0) -> None:
    health_url = api_base_url.rstrip("/") + "/health"
    chat_url = api_base_url.rstrip("/") + "/chat"

    print("Recipe Chatbot (CLI)")
    print("Enter ingredients or a recipe question (e.g. 'Egg, Onion'). Type 'quit' or 'exit' to stop.\n")

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(health_url)
            r.raise_for_status()
            data = r.json()
            print(f"Connected to API (model: {data.get('model', 'unknown')}).\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye.")
                    break

                try:
                    response = client.post(
                        chat_url,
                        json={"message": user_input},
                    )
                    response.raise_for_status()
                    data = response.json()
                    print("Bot:", data.get("response", ""))
                except httpx.HTTPStatusError as e:
                    print(f"Bot: [Error] API returned {e.response.status_code}: {e.response.text}", file=sys.stderr)
                except Exception as e:
                    print(f"Bot: [Error] {e}", file=sys.stderr)
                print()
    except httpx.HTTPError as e:
        print(f"Cannot reach API at {api_base_url}: {e}", file=sys.stderr)
        print("Make sure the API server is running (see README).", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recipe Chatbot CLI")
    parser.add_argument("--api-url", default=config.API_BASE_URL, help="API base URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
    args = parser.parse_args()
    chat_loop(api_base_url=args.api_url, timeout=args.timeout)


if __name__ == "__main__":
    main()
