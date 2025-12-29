import asyncio
from therapy_based_agent import TheraphyAgent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3
import os
def _print_messages(messages):
    if messages is None:
        return
    print("Agent:", messages[-1].content)


def _invoke(agent_compiled, state, config=None, context=None):
    return agent_compiled.invoke(state, config=config, context=context)


def main():
    os.makedirs('scratch/',exist_ok=True)

    check_conn = sqlite3.connect('scratch/therapy_langgraph_check.db', check_same_thread=False,isolation_level=None)
    mem_conn = sqlite3.connect('scratch/therapy_langgraph_mem.db', check_same_thread=False,isolation_level=None)

    compiled = TheraphyAgent().compile(checkpointer=SqliteSaver(check_conn), store=SqliteStore(mem_conn))

    user_id = input('Enter User ID: ')
    designation = input('Enter designation: ')

    print("(press Ctrl-C to exit)")
    try:
        while True:
            user_input = input('> ')
            state = {'messages': [HumanMessage(content=user_input)]}
            config = {"configurable": {"thread_id": 8, "user": user_id}}
            context = {"designation": designation, "user": user_id}

            result = _invoke(compiled, state, config=config, context=context)

            _print_messages(result.get("messages"))
            if result.get("key_points"):
                print("Key points:", result.get("key_points"))
            if result.get("store_memory"):
                print("(Agent requested to store memory)")

    except KeyboardInterrupt:
        check_conn.close()
        mem_conn.close()
        print("\nExiting.")


if __name__ == "__main__":
    main()
