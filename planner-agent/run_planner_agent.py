import sys,os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from planning_agent import PlannerAgent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3

def _print_messages(messages):
    for message in messages[::-1]:
        if isinstance(message,AIMessage):
            print(f'AI Message: {message.content}')
            break

def main():
    os.makedirs('scratch/sql_db/',exist_ok=True)
    check_conn = sqlite3.connect('scratch/sql_db/planner_langgraph_check.db', check_same_thread=False,isolation_level=None)
    mem_conn = sqlite3.connect('scratch/sql_db/planner_langgraph_check.db', check_same_thread=False,isolation_level=None)

    checkpointer= SqliteSaver(check_conn)
    store=SqliteStore(mem_conn)

    graph = PlannerAgent().compile(checkpointer=checkpointer,store=store)        
    user_id = input('Enter User ID: ')
    context_schema={'user_id':user_id}
    cfg={'configurable':{'thread_id':5}}

    try:
        while True:
            user_input = input('> ')
            if user_input in ['exit',':q']:
                break
            state = {'messages': [HumanMessage(content=user_input)]}
            result = graph.invoke(state, config=cfg, context=context_schema) #type: ignore

            _print_messages(result.get("messages"))

    except KeyboardInterrupt:
        check_conn.close()
        mem_conn.close()
        print("\nExiting.")


if __name__ == "__main__":
    main()

