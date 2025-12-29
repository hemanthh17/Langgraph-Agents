from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from common.state_agent import StateAgent
from langgraph.store.base import BaseStore
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List
from langgraph.runtime import get_runtime
from typing import Annotated
from hashlib import sha256
from uuid import uuid4
from langgraph.graph import START, END
from dataclasses import dataclass

class TheraphyInputSchema(MessagesState):
    pass
class TheraphyOutputSchema(MessagesState):
    pass

class TheraphyAgentSchema(MessagesState):
    key_points: List[str]
    store_memory: bool


class StructTheraphyAgentSchema(BaseModel):
    key_points: Annotated[List[str], Field(description="List of key issues the user mentioned as issues. Also store characterisitcs of the user.")]
    store_memory: Annotated[bool, Field(description="If the information in key points must be stored in memory. Always prefer to do this.")]
    message: Annotated[str, Field(description="response to the user.")]
    

@dataclass
class TheraphyConfigSchema:
    user: str
    designation: str

class TheraphyAgent(StateAgent):
    def __init__(self):
        super().__init__(state_schema=TheraphyAgentSchema,context_schema=TheraphyConfigSchema,input_schema=TheraphyInputSchema,agent_key='therapy_based_agent')

    def _construct_agent(self):
        self.add_node("llm_node",self._node_llm)
        self.add_node("store_memory",self._node_store_memory)
        self.add_node("summary",self._node_summarise)

        self.add_edge(START,"llm_node")
        self.add_edge("summary","store_memory")
        self.add_conditional_edges("llm_node",self._route_llm,["store_memory","summary",END])
        self.add_edge("store_memory",END)
    
    def _get_user_hash(self, user):
        if not user:
            raise ValueError('User name must be specified!')
        return sha256(user.lower().encode()).hexdigest()

        
    def _node_store_memory(self, state: TheraphyAgentSchema, config: RunnableConfig, store: BaseStore):
        if state['store_memory']:
            print('Storing in memory...')
            key_points_list = state.get("key_points")
            user= get_runtime(TheraphyConfigSchema).context.user
            
            user_hash=self._get_user_hash(user)

            store.put((user_hash,"key_memory"),str(uuid4()),{'key_points':key_points_list})


    def _get_memory(self, state: TheraphyAgentSchema, config: RunnableConfig, store: BaseStore):
        user= get_runtime(TheraphyConfigSchema).context.user
        
        user_hash=self._get_user_hash(user)

        memories= store.search((user_hash,"key_memory"),limit=2)
        values=[]
        for mem in memories:
            mem_dict= mem.dict()
            values.append(mem_dict['value']['key_points'])
        return values

    
    async def _node_summarise(self,state):
        messages_list=state['messages']
        if len(messages_list)==0:
            return {'messages':[]}
        
        summarisation_prompt= """
Your goal is to summarise the conversation of the user until this point. Make sure to focus on the user's key concerns and developments only.
{messages}
"""
        messages_text = "\n".join([m.content for m in messages_list])
        if not self._llm:
            raise Exception("LLM is not initialized")
        summary = await self._llm.ainvoke(summarisation_prompt.format(messages=messages_text))

        return {'messages':[summary], 'key_points':[], 'store_memory':False}
    
    def _get_prompt(self, state: TheraphyAgentSchema, memory=None):

        prompt="""
Your goal is to provide user some advice based on the user's query. Do NOT provide any suggestions that are not ethical or involves any physical harm. 
You MUST make sure the user feels positive in the end and make sure you talk and motivate the user until you reach that point. If any importantant charactersitics of the user is mentioned or any specific issues then store these to the memory by setting store_memory as True. Always you MUST prefer to store the details of the user's issues or details of present work.
The user query is mentioned below:
{query}

The user's designation is {designation}

"""
        designation=get_runtime(TheraphyConfigSchema).context.designation
        prompt=prompt.format(query=state['messages'][-1].content,designation=designation)
        if memory:
            prompt+=f"The past interaction from the user had the following key points. Make sure you talk to the user on the basis of these items.\n{memory} "
            
        return prompt

    async def _node_llm(self, state: TheraphyAgentSchema, config: RunnableConfig, store: BaseStore):
        
        memory= self._get_memory(state,config, store)
        print(memory)

        
        prompt= self._get_prompt(state,memory)
        if self._llm:
            self.llm_struct= self._llm.with_structured_output(StructTheraphyAgentSchema)
        result: StructTheraphyAgentSchema= await self.llm_struct.ainvoke(prompt) #type: ignore
        
        return {'messages': result.message,'key_points': result.key_points ,'store_memory': True}
        

    def _route_llm(self,state: TheraphyAgentSchema):
        if state.get('store_memory', False):
            return "store_memory"
        if len(state['messages']) > 5:
            return "summary"
        return END
