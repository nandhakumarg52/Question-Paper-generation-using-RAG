import os
import streamlit as st
from dotenv import load_dotenv
from question_paper_generator import VectorStoreManager, QGen
from llm import LLModel  # Assuming this is your language model class

# Load environment variables
load_dotenv()

def main():
    st.title("SAMBA PUBLICATIONS -**Question Paper Generator**")
    st.write("Upload a PDF, enter a query, and specify marks distribution to generate a custom question paper.")

    # File uploader for the PDF
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Input fields for the query
    query = st.text_input("Enter the query")

    # Marks distribution using checkboxes
    st.write("Select the marks distribution:")
    
    # Define marks and their corresponding count inputs
    marks_distribution = {
        1: st.checkbox("1 mark multiple choice questions"),
        2: st.checkbox("2 mark short answer questions"),
        3: st.checkbox("3 mark short answer questions"),
        5: st.checkbox("5 mark long answer questions"),
        10: st.checkbox("10 mark long answer questions")
    }

    # Count input for each mark type
    counts = {}
    for mark, selected in marks_distribution.items():
        if selected:
            counts[mark] = st.number_input(f"How many {mark} mark questions?", min_value=1, step=1, key=f"count_{mark}")

    # Button to trigger question paper generation
    if st.button("Generate Question Paper"):
        if not pdf_file:
            st.error("Please upload a PDF file.")
            return

        if not query or not counts:
            st.error("Please provide the query and marks distribution.")
            return

        # Ensure the temp directory exists
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the uploaded file temporarily
        pdf_file_path = os.path.join(temp_dir, pdf_file.name)
        with open(pdf_file_path, "wb") as f:
            f.write(pdf_file.read())

        # Generate the question pattern based on user input
        pattern_strings = []
        for mark, count in counts.items():
            if mark == 1:
                pattern_strings.append(f"{count} multiple choice questions (each carries {mark} mark)")
            elif mark == 2 or mark == 3:
                pattern_strings.append(f"{count} short answer questions (each carries {mark} marks)")
            elif mark >= 5:
                pattern_strings.append(f"{count} long answer questions (each carries {mark} marks)")
            else:
                st.error(f"Unexpected mark value: {mark}.")
                return

        if not pattern_strings:
            st.error("No valid question patterns provided.")
            return

        question_pattern = (
            f"Draft a {query}. The question paper should include " +
            ", ".join(pattern_strings) +
            ". Ensure the questions aren't out of the syllabus" +
            " (DO NOT USE ONLINE KNOWLEDGE OTHER THAN RETRIEVER AND NEVER GIVE ANSWER JUST GIVE QUESTION)."
        )

        # Load environment variables for Qdrant and Hugging Face API keys
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        hf_token = os.getenv("HF_API_KEY")

        # Step 1: Initialize VectorStoreManager with the uploaded PDF
        vectorstore_manager = VectorStoreManager(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            hf_token=hf_token,
            embedding_model_name="hf",
            model_name="sentence-transformers/all-roberta-large-v1"
        )

        # Step 2: Process the PDF to create a vector store
        vectorstore = vectorstore_manager.process_file(file_path=pdf_file_path)

        # Step 3: Initialize the language model (assuming 'LLModel' is a predefined class for handling LLMs)
        llm_model = LLModel.initialize_llm(model_name="groq")  # Replace with the actual method to load your model

        # Step 4: Initialize QGen with the vector store, language model, and question pattern
        qg = QGen(vectorstore=vectorstore, llm=llm_model, question_pattern=question_pattern)

        # Step 5: Generate the question paper
        result = qg.generate_question_paper()

        # Display the result
        st.subheader("Generated Question Paper:")
        st.write(result)

if __name__ == "__main__":
    main()
