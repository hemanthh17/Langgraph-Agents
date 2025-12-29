from langgraph.graph import StateGraph
from abc import ABCMeta, abstractmethod
from common.llm_utils import LLM
from langchain_core.language_models import BaseChatModel
from typing import Optional

class StateAgent(StateGraph,metaclass=ABCMeta):
    def __init__(self,state_schema, input_schema=None, output_schema=None, context_schema=None, llm_object: Optional[BaseChatModel]=None,
                 agent_key: Optional[str]=None):
        self._input_schema=input_schema
        self._output_schema=output_schema
        self._state_schema=state_schema
        self._context_schema=context_schema
        # Prioritising LLM Object over the cofig key
        if llm_object is not None:
            self._llm=llm_object
        elif agent_key is not None:
            self._llm= LLM().get_llm(key=agent_key)
        else:
            raise Exception('Configuring LLM requires either the LLM object or the agent key in config.yaml')        

        super().__init__(state_schema=self._state_schema,context_schema=self._context_schema,input_schema=self._input_schema,output_schema=self._output_schema)

        self._construct_agent()

    @abstractmethod
    def _construct_agent(self):
        ...
    
    @property
    def llm(self):
        return self._llm
    










        
