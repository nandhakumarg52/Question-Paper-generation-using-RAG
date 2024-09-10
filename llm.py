import model
import warnings
warnings.filterwarnings("ignore") 
class LLModel:
    def __init__(self,model_name):
        self.model_name = model_name
    def initialize_llm( model_name):
        """
        Initializes the LLM model based on the provided model_name.

        Args:
            model_name (str): The name of the model to use ("groq", "openai", etc.).

        Returns:
            An instance of the selected language model.
        """
        if model_name == "groq":
            return model.get_groq_chat(model_name="llama3-70b-8192")  # Example Groq model
        elif model_name == "openai":
            return model.get_openai_chat(model_name="gpt-3.5-turbo")  # Example OpenAI model
        elif model_name == "azure_openai":
            return model.get_azure_openai_chat(deployment_name="your_deployment_name")  # Example Azure OpenAI model
        elif model_name == "anthropic":
            return model.get_anthropic_chat(model_name="claude-v1")  # Example Anthropic model
        elif model_name == "ollama":
            return model.get_ollama_chat(model_name="mistral")  # Example Ollama model
        elif model_name == "google":
            return model.get_google_chat(model_name="chat-bison")  # Example Google model
        elif model_name == "lmstudio":
            return model.get_lmstudio_chat(model_name="gptj")  # Example LM Studio model
        elif model_name == "openrouter":
            return model.get_openrouter(model_name="meta-llama/llama-3.1-8b-instruct:free")  # Example OpenRouter model
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
