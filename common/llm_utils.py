from dotenv import load_dotenv
import os 
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pathlib import Path
import yaml


class LLM:
    def __init__(self,settings_path=None):
        load_dotenv()
        if settings_path is None:
            settings_path=Path(__file__).resolve().parents[1] / "config.yaml"
        self.settings_path=Path(settings_path)

    def get_llm(self,llm_object=None,key=None):
        if llm_object is None and key is None:
            raise Exception('You will need to pass any one of the values. LLM object or key')
        if llm_object is not None:
            return llm_object
        if key is not None:
            with open(self.settings_path) as settings:
                settings=yaml.safe_load(settings)
                agents=settings.get('agents')
                agent_info=None
                for agent in agents:
                    if agent.get('name')==key:
                        agent_info=agent
                if agent_info is None:
                    raise Exception('No agent found in settings!!')
                model_provider=agent_info.get('model_provider')
                model_name=agent_info.get('model_name')
                timeout=agent_info.get('timeout',10)
                stop=agent_info.get('stop')
                temperature=agent_info.get('temperature',0)

            if model_provider=="ollama":
                return ChatOllama(model=model_name, temperature=temperature)
            elif model_provider=='openai':
                if not os.environ.get('OPENAI_API_KEY'):
                    if os.environ.get("OPENAI_API_KEY") is None:
                        raise Exception("OpenAI Key must be provided! Do Not mention the API key in config.yaml")
                return ChatOpenAI(name=model_name, temperature=temperature)
            elif model_provider=="anthropic":
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    if os.environ.get("ANTHROPIC_API_KEY") is None:
                        raise Exception("Anthropic Key must be provided! Do Not mention the API key in config.yaml")
                return ChatAnthropic(model_name=model_name,timeout=timeout,stop=stop, temperature=temperature)
            else:
                raise Exception("Model Provider not defined! This is future work")
