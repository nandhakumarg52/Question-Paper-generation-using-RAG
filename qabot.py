import os
# from crewai import Agent, Task, Crew, Process 
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
# from crewai_tools import BaseTool
from typing import Optional, Type, Any
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

class QAChatBot:
    def __init__(self, vectorstore,llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.qa_chain = self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        template = """
# System Prompt for Generating Question Papers

You are an AI system designed to generate examination question papers. Use the following guidelines to create a new question paper for any subject. Ensure that the paper includes a balance of multiple-choice questions (MCQs), short-answer questions (SAQs), long-answer questions (LAQs), and other question types like matching, fill-in-the-blank, and comprehension-based questions.

## Instructions for generating a question paper:

### 1. Title Section:
- Clearly mention the subject name and term (e.g., "English Term 1").

### 2. MCQs (Multiple Choice Questions):
- Include 5–10 multiple-choice questions.
- For each question, provide 3 options (a, b, c).
- **Example format:**
    ```
    1. Kavin _________ a pineapple yesterday.
        a) eats
        b) ate
        c) is eating
    ```

### 3. Fill-in-the-Blank Questions:
- Include 5 fill-in-the-blank questions where students provide a missing word.
- **Example format:**
    ```
    I brush my __________ (tooth) daily.
    ```

### 4. Plural Forms:
- Provide words where students have to write the plural forms (3–5 items).
- **Example:**
    ```
    Three __________ (woman) were talking in the park.
    ```

### 5. Short Answer Questions (SAQs):
- Include 3–5 questions that require 2-3 sentence answers.
- **Example format:**
    ```
    Write two sentences about your favorite fruit.
    ```

### 6. Long Answer Questions (LAQs):
- Include 2–3 long-answer questions that require more elaborate responses.
- **Example:**
    ```
    Write a letter to your friend inviting them for the holidays.
    ```

### 7. Comprehension:
- Include a short passage followed by 3–5 questions related to it.
- **Example format:**
    ```
    Read the following passage and answer the questions below:
    [Insert passage here]
    ```

### 8. Miscellaneous:
- Include sections for matching, opposite words, and correct word usage (e.g., buy vs. bye).
- **Example format:**
    ```
    Match the following words with their antonyms.
    ```

Ensure the generated question paper is logically structured and covers a wide range of question formats to assess different skills.

        """
        retriever = self.vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_template(template)
        handler = StdOutCallbackHandler()

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            callbacks=[handler],
            retriever=retriever
        )

    def run(self, question: str) -> str:
        if not question:
            return "No question provided"
        try:
            result = self.qa_chain.invoke(question)
            if result is None:
                return "Failed to retrieve an answer. No context available."
            return result
        except Exception as e:
            return f"Error during question processing: {e}"


class RetrieverToolSchema(BaseModel):
    """Input for RetrieverTool."""
    question: str = Field(
        ...,
        description="The question to retrieve an answer for using the QAChatBot."
    )

class RetrieverTool(BaseTool):
    name: str = "Text book"
    description: str = "Elementary school textbook"
    args_schema: Type[BaseModel] = RetrieverToolSchema
    qa_chatbot: Optional[Any] = None

    def __init__(self,qa_chatbot: Optional[Any] = None,**kwargs):
        super().__init__(**kwargs)
        if qa_chatbot is not None:
            self.qa_chatbot = qa_chatbot
            self.description = "A tool that can be used to retrieve information from the vector store using the QAChatBot."
            self.args_schema = RetrieverToolSchema
            # self._generate_description()

    def _run(self, **kwargs: Any) -> Any:
        try:
            question = kwargs.get('question')
            if question is None:
                return "No question provided."
            return self.qa_chatbot.run(question)
        except Exception as e:
            return f"Failed to retrieve information. Error: {e}"

class QGen:
    """
    Manages the  AI agents and tasks for retrieving information and generating question papers.
    """
    def __init__(self, vectorstore, llm, question_pattern=None):
        self.llm = llm

        # Initialize QAChatBot with the provided vectorstore and language model
        self.qa_chatbot = QAChatBot(vectorstore=vectorstore, llm=self.llm)

        # Initialize the Retriever Tool with the QAChatBot instance
        self.rag_tool = RetrieverTool(qa_chatbot=self.qa_chatbot)

        self.question = question_pattern

    def generate_question_paper(self):
        """
        Kicks off the Crew AI process to generate a question paper.

        Args:
            question (str): The question to be processed by the agents.

        Returns:
            str: The result of the Crew AI process.
        """
        inputs = {"question": self.question}
        res = self.rag_tool._run(**inputs)
        return str(res['result'])
