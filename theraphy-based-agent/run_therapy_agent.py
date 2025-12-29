import asyncio
from therapy_based_agent import TheraphyAgent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph
from langchain_core.messages import HumanMessage, AIMessage

def _print_messages(messages):
    if messages is None:
        return
    print("Agent:", messages[-1].content)


async def _ainvoke(agent_compiled, state, config=None, context=None):
    return await agent_compiled.ainvoke(state, config=config, context=context)


async def main_async():
    compiled = TheraphyAgent().compile(checkpointer=InMemorySaver(), store=InMemoryStore())

    user_id = input('Enter User ID: ')
    designation = input('Enter designation: ')

    print("(press Ctrl-C to exit)")
    try:
        while True:
            user_input = input('> ')
            state = {'messages': [HumanMessage(content=user_input)]}
            config = {"configurable": {"thread_id": 3, "user": user_id}}
            context = {"designation": designation, "user": user_id}

            result = await _ainvoke(compiled, state, config=config, context=context)

            _print_messages(result.get("messages"))
            if result.get("key_points"):
                print("Key points:", result.get("key_points"))
            if result.get("store_memory"):
                print("(Agent requested to store memory)")

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    asyncio.run(main_async())
