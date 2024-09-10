from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from vectorstore_manager import VectorStoreManager
from qabot import QGen
from llm import LLModel
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/generate-question-paper', methods=['POST'])
def generate_question_paper():
    try:
        load_dotenv()

        # Check if a file part is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']

        # Get the title or query (i.e., the subject or title of the question paper)
        query = request.form.get('title')
        print(file)
        print(query)

        # Collect all marks and their corresponding counts dynamically
        pattern_strings = []
        for key, value in request.form.items():
            if key.endswith('mark') and value.isdigit():
                mark = key.replace('mark', '').strip()  # Get the mark value by removing 'mark' from the key
                count = value.strip()  # Get the count (number of questions)
                if mark.isdigit():
                    mark = int(mark)  # Convert mark to an integer for comparison
                    if mark == 1:
                        pattern_strings.append(f"{count} multiple choice questions (each carries {mark} mark)")
                    elif mark == 2 or mark == 3:
                        pattern_strings.append(f"{count} short answer questions (each carries {mark} marks)")
                    elif mark > 3:
                        pattern_strings.append(f"{count} long answer questions (each carries {mark} marks)")
                    else:
                        return jsonify({"error": f"Unexpected mark value: {mark}"}), 400

        # Check if a valid file was uploaded
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Check if the query or title is provided
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Check if valid question patterns were provided
        if not pattern_strings:
            return jsonify({"error": "No valid question patterns provided"}), 400

        # Combine all parts into one string for the question pattern
        question_pattern = (
            f"Draft a question paper by analyzing the retrieved content. The question paper should include " +
            ", ".join(pattern_strings) +
            ". Ensure the questions aren't out of the syllabus (DO NOT USE ONLINE KNOWLEDGE OTHER THAN RETRIEVER)."
        )

        # Save the uploaded file to the 'data' directory
        os.makedirs("data", exist_ok=True)
        pdf_file_path = os.path.join("data", file.filename)
        file.save(pdf_file_path)

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
            model_name="sentence-transformers/all-mpnet-base-v2"  # Pass the model name
        )

        # Step 2: Initialize vector store using Qdrant
        vectorstore = vectorstore_manager.process_file(file_path=pdf_file_path)

        # Step 3: Initialize the language model
        llm_model = LLModel.initialize_llm(model_name="groq")  # Replace with desired LLM

        # Step 4: Initialize QGen with the vector store, LLM, and question pattern
        qg = QGen(vectorstore=vectorstore, llm=llm_model, question_pattern=question_pattern)

        # Step 5: Generate the question paper using the QGen manager
        result = qg.generate_question_paper()

        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)
