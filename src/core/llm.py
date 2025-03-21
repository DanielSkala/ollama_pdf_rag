"""LLM configuration and setup."""

import logging

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.chat_models import ChatOllama

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM configuration and prompts."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)

    def get_query_prompt(self) -> PromptTemplate:
        """Get query generation prompt."""
        return PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

    def get_rag_prompt(self) -> ChatPromptTemplate:
        """Get RAG prompt template."""
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)


if __name__ == "__main__":
    llm_manager = LLMManager()
    query_prompt = llm_manager.get_query_prompt()
    rag_prompt = llm_manager.get_rag_prompt()

    print(f"Query prompt: {query_prompt}\n")
    print(f"RAG prompt: {rag_prompt}")

    # Basic chatting
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm_manager.llm.invoke(messages)
    print(ai_msg.content)
