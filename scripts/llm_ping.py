#!/usr/bin/env python3
import os
import sys
import ssl
import argparse


def main():
    ap = argparse.ArgumentParser(description="Minimal OpenAI chat ping to validate API key and TLS")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    args = ap.parse_args()

    print("Python:", sys.version.split()[0])
    print("SSL:", ssl.OPENSSL_VERSION)

    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    print("OPENAI_API_KEY set:", key_present)
    if not key_present:
        print("ERROR: OPENAI_API_KEY not found in environment")
        sys.exit(2)

    try:
        # Prefer new OpenAI client if available
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            print("OpenAI client: openai>=1.0 detected")
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with exactly: ping-ok"},
                ],
                max_tokens=4,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            print("Response:", text)
            if text.lower().startswith("ping-ok"):
                print("STATUS: SUCCESS")
                sys.exit(0)
            else:
                print("STATUS: UNEXPECTED_CONTENT")
                sys.exit(3)
        except ImportError:
            import openai  # type: ignore
            print("OpenAI client: legacy openai detected")
            openai.api_key = os.environ["OPENAI_API_KEY"]
            resp = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with exactly: ping-ok"},
                ],
                max_tokens=4,
                temperature=0,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            print("Response:", text)
            if text.lower().startswith("ping-ok"):
                print("STATUS: SUCCESS")
                sys.exit(0)
            else:
                print("STATUS: UNEXPECTED_CONTENT")
                sys.exit(3)
    except Exception as e:
        print("STATUS: ERROR")
        print(type(e).__name__ + ":", e)
        sys.exit(1)


if __name__ == "__main__":
    main()


