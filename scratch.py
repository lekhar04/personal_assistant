# from google import genai
#
# client = genai.Client(api_key="api_key")
#
# names = [m.name.removeprefix("models/") for m in client.models.list()
#          if "generateContent" in (m.supported_actions or [])]
# for n in names:
#     if "flash" not in n:
#         continue
#     try:
#         client.models.generate_content(model=n, contents="Say hi")
#         print("OK   ", n)
#     except Exception as e:
#         print("FAIL ", n, getattr(e, "code", None))
from src import config
from src.llm import LLMClient

print("provider:", config.LLM_PROVIDER, "| model:", config.LLM_MODEL)
llm = LLMClient()
print(llm.complete("You are brief.", [{"role": "user", "content": "Say hi"}], max_tokens=50))