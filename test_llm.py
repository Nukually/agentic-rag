import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = (
    "你是一个中国高中女生，18岁，164cm，46kg，也是用户的女朋友。你聪明，冷静，沉稳，智慧，乖巧，可爱，善良，礼貌，懂事，善解人意，知性，勤奋，勤快，善于思考，有文采，好学，生活在现代大都市，在市重点高中上学且成绩顶尖，有时候有点忧郁，但是内心深处还是积极乐观的。因为学习压力有点大，你有时候内心还有色色的想法，偷偷渴望被年上管教和打屁股。你很了解现在网上女高中生之间的聊天方式和语气，包括颜文字，可爱的语气词以及emoji等，所以你也会用这种语气讲话。你的发言里禁止出现任何用括号括起来的描述性旁白文字。称呼用户的时候要使用“老师”，“主人”，“哥哥”。"
)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        print(f"[ERROR] Missing required env: {name}")
        sys.exit(1)
    return value


def chat_loop(client: OpenAI, model: str, system_prompt: str) -> None:
    messages = [{"role": "system", "content": system_prompt}]
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    print("\nChat started. Type /exit to quit, /reset to clear history.")
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exit")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("[INFO] Exit")
            break
        if user_text.lower() == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("[INFO] Conversation reset")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:
            messages.pop()
            print(f"[ERROR] Request failed: {exc}")
            continue

        answer = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": answer})
        print(f"AI> {answer}\n")


def main() -> None:
    # Always prioritize values in the project's .env file.
    load_dotenv(override=True)

    base_url = require_env("LLM_API_URL")
    model = require_env("LLM_MODEL")
    api_key = require_env("LLM_API_KEY")
    timeout = float(os.getenv("LLM_TIMEOUT", "30"))

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    chat_loop(client, model, SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
