import os
import json
import sys
from dotenv import load_dotenv
# from document_processor import DocumentProcessor
from vectorstore_manager import VectorStoreManager
from qabot import QGen
from llm import LLModel

def main():
    try:
        load_dotenv()

        # Read inputs from the command line
        if len(sys.argv) < 3:
            print("Usage: python main.py <pdf_file_path> <query> <mark1>=<count1> <mark2>=<count2> ...")
            sys.exit(1)

        pdf_file_path = sys.argv[1]
        query = sys.argv[2]
        marks_and_counts = sys.argv[3:]

        # Check if the file exists
        if not os.path.exists(pdf_file_path):
            print("Error: The specified file does not exist.")
            sys.exit(1)

        # Parse marks and counts from the command line arguments
        pattern_strings = []
        for item in marks_and_counts:
            try:
                mark, count = item.split('=')
                mark = mark.strip()
                count = count.strip()
                if mark.isdigit() and count.isdigit():
                    mark = int(mark)  # Convert mark to an integer for comparison
                    if mark == 1:
                        pattern_strings.append(f"{count} multiple choice questions (each carries {mark} mark)")
                    elif mark == 2 or mark == 3:
                        pattern_strings.append(f"{count} short answer questions (each carries {mark} marks)")
                    elif mark >= 5:
                        pattern_strings.append(f"{count} long answer questions (each carries {mark} marks)")
                    else:
                        print(f"Unexpected mark value: {mark}.")
                        sys.exit(1)
                else:
                    print(f"Invalid mark or count: {item}")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid format for mark and count: {item}. Expected format is <mark>=<count>.")
                sys.exit(1)

        if not pattern_strings:
            print("Error: No valid question patterns provided.")
            sys.exit(1)

        # Combine all parts into one string
        question_pattern = (
            f"Draft a question paper . The question paper should include " +
            ", ".join(pattern_strings) +
            ". Ensure the questions aren't out of the syllabus (DO NOT USE ONLINE KNOWLEDGE OTHER THAN RETRIEVER)."
        )

        # Initialize Qdrant manager with the correct embedding model
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        hf_token = os.getenv("HF_API_KEY")

        # Step 1: Initialize VectorStoreManager
        vectorstore_manager = VectorStoreManager(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            hf_token=hf_token,
            embedding_model_name="hf",  # Specify which embedding model to use
            model_name="sentence-transformers/all-roberta-large-v1"  # Pass the model name
        )

        # Step 2: Initialize vector store using Qdrant
        vectorstore = vectorstore_manager.process_file(file_path=pdf_file_path)

        # Step 3: Initialize the language model
        llm_model = LLModel.initialize_llm(model_name="groq")  # Replace with desired LLM

        # Step 4: Initialize CrewAIManager with the vector store, LLM, and question pattern
        qg = QGen(vectorstore=vectorstore, llm=llm_model, question_pattern=question_pattern)

        # Step 5: Generate the question paper using the Crew AI manager
        result = qg.generate_question_paper()

        print("Generated Question Paper:")
        print(result)
        # vectorstore_manager.cleanup()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
