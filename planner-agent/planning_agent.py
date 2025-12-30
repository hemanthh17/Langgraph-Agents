import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from langgraph.graph import MessagesState, START, END
from common.state_agent import StateAgent
from pydantic import BaseModel, Field
from typing import Annotated, List
from langgraph.types import Overwrite
from langgraph.store.base import BaseStore
from langgraph.runtime import get_runtime
from hashlib import sha256
from uuid import uuid4
from langchain_core.messages import AIMessage

class PlannerInputSchema(MessagesState):
    pass 

class PlannerStateSchema(MessagesState):
    solutions_list: List[str]
    store_memory: bool

class PlannerContextSchema(BaseModel):
    user_id: str

class PlannerOutputSchema(MessagesState):
    pass

class StructPlannerSchema(BaseModel):
    message: Annotated[str, Field(description="The message to provide to the user")]
    solutions_list: Annotated[List[str], Field(description="The list of solutions to user's problem.")]
    store_memory: Annotated[bool,Field(description="Should the key solutions list be stored to the memory")]


class PlannerAgent(StateAgent):
    def __init__(self,llm_object=None,agent_key="planning_agent"):
        super().__init__(state_schema=PlannerStateSchema,input_schema=PlannerInputSchema,output_schema=PlannerOutputSchema,context_schema=PlannerContextSchema,llm_object=llm_object,agent_key=agent_key)

        if self._llm:
            self.llm_struct=self._llm.with_structured_output(StructPlannerSchema)
        
    
    def _construct_agent(self):
        self.add_node("summarise",self._node_summarise)
        self.add_node("llm_node",self._node_llm)
        self.add_node("store_memory",self._node_store_memory)

        self.add_edge(START,"llm_node")
        self.add_conditional_edges("llm_node",self._route_llm,["summarise","store_memory",END])

    def _get_user_hash(self):
        user_id = get_runtime(PlannerContextSchema).context.user_id
        hashed_user= sha256(user_id.encode()).hexdigest()

        return hashed_user
    
    def _node_store_memory(self, state:PlannerStateSchema, store: BaseStore):
        print('Storing in memory...')
        user_hash= self._get_user_hash()
        memory_namespace=(user_hash,"plan_steps")
        store.put(memory_namespace,str(uuid4()),{'plan_steps':state["solutions_list"]})

    def _get_memory(self,store:BaseStore):
        user_hash= self._get_user_hash()
        memory_namespace=(user_hash,"plan_steps")
        memories= store.search(memory_namespace)
        values=[]
        for memory in memories:
            values.append(memory.dict()['value']['plan_steps'])
        return values
    
    def _node_summarise(self, state: PlannerStateSchema):
        summary_prompt=f"You must make sure to summarise the key elements in the conversation. Make sure to focus on summarising only the key issues the user mentioned.\n{'\n'.join([str(m.content) for m in state['messages']])}"

        if self._llm:
            summary=self._llm.invoke(summary_prompt)
            return {'messages':Overwrite(AIMessage(content=str(summary)))}
        
    
    def _get_prompt(self, state: PlannerStateSchema, store: BaseStore):
        user_name = get_runtime(PlannerContextSchema).context.user_id
        prompt="""
Your goal is to make sure you list out the steps to address and solve user's issues. Based on the user prompt make sure you follow the schema and provide a list of achievable and logical list of actions the user can do to solve the issue. The user details are mentioned below along with the prompt. Address the user with their user name.

User Name: {user_name}

query:
{query}
"""
        prompt= prompt.format(user_name=user_name,query=state['messages'][-1].content)
        memory=self._get_memory(store)

        if memory:
            add_content= f"\nBased on the previous user interaction, the following set of action items were procured.\n{memory}"
            prompt+=add_content
        return prompt
    
    def _node_llm(self,state: PlannerStateSchema, store: BaseStore):
        prompt= self._get_prompt(state,store)
        result: StructPlannerSchema = self.llm_struct.invoke(prompt) #type: ignore

        return {'messages': [AIMessage(result.message)],'solutions_list':result.solutions_list,'store_memory':True}
    
    def _route_llm(self,state: PlannerStateSchema):
        if state.get('store_memory', False):
            return "store_memory"
        if len(state['messages']) > 10:
            return "summarise"
        return END













