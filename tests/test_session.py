from orchestrator.langchain_orchestrator import create_orchestrator_with_langchain
import uuid


session = uuid.uuid4().hex[:8]
print(f"Session ID: {session}")
orch = create_orchestrator_with_langchain()

print("Type /exit to quit.")
while True:
    q = input("You: ")
    if q.strip().lower() in {"/exit", "exit", "quit"}:
        break
    for chunk in orch.stream(q, session_id=session):
        print(f"Bot: {chunk}")