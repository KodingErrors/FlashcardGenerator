import sys
import os
import random
import tempfile
import shutil
import streamlit as st

osos.environ.get('GOOGLE_API_KEY')
os.environ.get('GOOGLE_CSE_ID')

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,    
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool 

try:
    from langchain_google_community.search import GoogleSearchAPIWrapper
except ImportError:
    st.error("Could not import GoogleSearchAPIWrapper or Out of API credits")
    st.stop() 


st.set_page_config(page_title="📚 Flashcard Generator", layout="wide")

st.title("📚 Flashcard Generator")
st.caption("Powered by Ollama + Langchain + Streamlit + Google Search (via tool)")

# --- Session State Management ---
# Temporary directory for storing uploaded files and FAISS index
if "db_dir" not in st.session_state:
    st.session_state.db_dir = tempfile.mkdtemp()

# List to store generated flashcards (question, correct_answer, [choices])
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
    st.session_state.wrong_answers = [] # To track incorrectly answered flashcards

# Session state for the quiz interface
if "current_flashcard_index" not in st.session_state:
    st.session_state.current_flashcard_index = 0
if "user_answer" not in st.session_state:
    st.session_state.user_answer = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False # Flag to indicate if results should be shown for the current question

def generate_flashcards_from_docs(docs):
    flashcards = []
    for doc in docs:
        content = doc.page_content.split("\n")
        for line in content:
            if line.strip():
                question = f"What is {line.strip()}?"
                answer = line.strip()
                choices = [answer] + generate_fake_choices_from_list(answer)
                random.shuffle(choices)
                while len(choices) < 4: # Ensure 4 choices
                    new_fake = f"Option {len(choices) + 1}"
                    if new_fake not in choices:
                        choices.append(new_fake)
                choices = choices[:4]
                flashcards.append((question, answer, choices))
    return flashcards

def generate_fake_choices_from_list(correct_answer):
    all_possible_fakes = ["Variable", "Function", "Class", "Module", "Loop", "Conditional",
                          "List", "Dictionary", "Tuple", "Set", "String", "Integer",
                          "Float", "Boolean", "Syntax Error", "Runtime Error",
                          "Algorithm", "Data Structure", "Framework", "Library"]
    fake_answers = [ans for ans in all_possible_fakes if ans.lower() != correct_answer.lower()]
    return random.sample(fake_answers, min(3, len(fake_answers)))


# --- LLM-based Flashcard Generation from Text ---
def generate_flashcards_with_llm(text_content: str, topic: str, num_flashcards: int = 5):
    llm = Ollama(model="llama3") # Ensure llama3 model is running

    # Prompt for the LLM to generate flashcards in a parsable format
    # We ask for JSON output for easier parsing.
    prompt_template = """You are an expert educator. Your task is to create multiple-choice flashcards based on the following text content about "{topic}".

Generate exactly {num_flashcards} flashcards. Each flashcard should consist of:
1. A clear question based on the text.
2. The correct answer from the text.
3. Exactly three plausible but incorrect multiple-choice options that are related to the topic but distinct from the correct answer.
4. The correct answer should be included among the options.

Provide the output as a JSON array of objects. Each object in the array should have the keys: "question", "correct_answer", and "options". The "options" key should contain a JSON array of four strings (the correct answer and three incorrect options).

Example JSON format:
[
    {{
        "question": "What is the main topic of the text?",
        "correct_answer": "Topic Name",
        "options": ["Topic Name", "Related Concept 1", "Related Concept 2", "Related Concept 3"]
    }},
    {{
        "question": "Which concept is defined as...?",
        "correct_answer": "Another Concept",
        "options": ["Wrong Answer A", "Another Concept", "Wrong Answer B", "Wrong Answer C"]
    }}
]

Ensure the JSON is correctly formatted and can be parsed directly. Do not include any other text or explanation outside the JSON array.

Text Content:
---
{text_content}
---

JSON Flashcards:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text_content", "topic", "num_flashcards"]
    )

    chain = prompt | llm

    response_text = "" 
    try:
        # Invoke the LLM and parse the JSON output
        response_text = chain.invoke({"text_content": text_content, "topic": topic, "num_flashcards": num_flashcards})
        
        import json
        
        if response_text.strip().startswith("```json"):
             response_text = response_text.strip()[len("```json"):].strip()
             if response_text.endswith("```"):
                 response_text = response_text[:-len("```")].strip()

        flashcards_data = json.loads(response_text)

        
        flashcards = []
        for item in flashcards_data:
            # Relax the initial validation - just check for essential keys and options being a list
            if "question" in item and "correct_answer" in item and "options" in item and isinstance(item["options"], list):
                question = item["question"]
                correct_answer = item["correct_answer"]
                options = item["options"][:] # Create a copy to modify

                # Ensure correct answer is in options (case-insensitive check)
                # If not, add it.
                if correct_answer.lower().strip() not in [opt.lower().strip() for opt in options]:
                    options.append(correct_answer)
                    
                seen_options = {}
                unique_options = []
                for opt in options:
                    clean_opt = opt.lower().strip()
                    if clean_opt not in seen_options:
                        seen_options[clean_opt] = opt 
                        unique_options.append(opt)

                options = unique_options 


                if len(options) > 4:
                    random.shuffle(options)
                    options = options[:4]

                while len(options) < 4:
                    placeholder = f"Option {len(options) + 1}"
                    # Ensure placeholder isn't accidentally a duplicate (unlikely but safe check)
                    if placeholder.lower().strip() not in seen_options:
                         options.append(placeholder)
                         seen_options[placeholder.lower().strip()] = placeholder # Add to seen to avoid adding same placeholder
                    else: # If placeholder is already in options, try a different one
                         # This case is highly unlikely for simple "Option X" placeholders
                         placeholder = f"Placeholder {len(options) + 1}"
                         if placeholder.lower().strip() not in seen_options:
                             options.append(placeholder)
                             seen_options[placeholder.lower().strip()] = placeholder
                         else:
                              # If we somehow can't generate a unique placeholder, break to avoid infinite loop
                              break


                # Final safeguard: Ensure the correct answer is still among the final 4 options
                if correct_answer.lower().strip() not in [opt.lower().strip() for opt in options]:
                     # If trimmed out, replace a random incorrect option with the correct one
                     incorrect_options = [opt for opt in options if opt.lower().strip() != correct_answer.lower().strip()]
                     if incorrect_options:
                          option_to_replace = random.choice(incorrect_options)
                          replace_index = options.index(option_to_replace)
                          options[replace_index] = correct_answer
                     else: # This should ideally not happen if correct answer was added earlier
                         st.error(f"Failed internal check: Correct answer '{correct_answer}' not in final options for question '{question}'. Options: {options}")
                         
                         continue 
                         
                random.shuffle(options)
                flashcards.append((question, correct_answer, options))
            else:
                st.warning(f"Skipping invalid flashcard data from LLM: {item} - Missing required keys or 'options' is not a list.")


        return flashcards

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from LLM response: {e}")
        st.text_area("LLM Response (Non-JSON):", response_text) # Show LLM output for debugging
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during flashcard generation: {e}")
        st.text_area("LLM Response (for debugging):", response_text) # Show LLM output for debugging
        return []
        
        
# --- Sidebar for File Upload and Internet Generation ---
with st.sidebar:
    st.header("📂 Upload Files")
    uploaded_files = st.file_uploader("Choose files (PDF, DOCX, PPTX, EPUB, TXT)",
                                         type=["pdf", "docx", "pptx", "epub", "txt"], # Note: EPUB is still listed here, but will be ignored in the loading logic below
                                         accept_multiple_files=True)

    if st.button("📥 Index Files"):
        if uploaded_files:
            docs = []
            with st.spinner("Loading and processing files..."):
                for file in uploaded_files:
                    suffix = os.path.splitext(file.name)[1].lower()
                    path = os.path.join(st.session_state.db_dir, file.name)
                    try:
                        with open(path, "wb") as f:
                            f.write(file.getbuffer())

                        loader = None
                        if suffix == ".pdf":
                            loader = PyMuPDFLoader(path)
                        elif suffix == ".txt":
                            loader = TextLoader(path)
                        elif suffix == ".docx":
                            loader = Docx2txtLoader(path)
                        elif suffix == ".pptx":
                            loader = UnstructuredPowerPointLoader(path)
                        
                        else:
                            st.warning(f"Unsupported file type: {suffix}")
                            continue # Skip to the next file if type is unsupported (including epub now)

                        if loader:
                            docs.extend(loader.load())
                    except Exception as e:
                        st.error(f"Error processing file {file.name}: {e}")


            if docs:
                with st.spinner("Splitting text and creating embeddings..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(docs)
                    try:
                        embeddings = OllamaEmbeddings(model="llama3") # Ensure llama3 model is running
                        db = FAISS.from_documents(texts, embeddings)
                        db.save_local(st.session_state.db_dir)
                    except Exception as e:
                        st.error(f"Error creating embeddings or FAISS index. Make sure Ollama is running and model 'llama3' is downloaded. Error: {e}")
                        st.stop() # Stop execution if embeddings fail

                # Generate flashcards from documents using the basic function
                with st.spinner("Generating flashcards from documents..."):
                    flashcards = generate_flashcards_from_docs(docs)
                    st.session_state.flashcards = flashcards
                    st.session_state.current_flashcard_index = 0 # Start quiz from the beginning
                    st.session_state.user_answer = None
                    st.session_state.show_results = False
                    st.session_state.wrong_answers = [] # Reset wrong answers for a new set

                st.success(f"Files indexed and {len(flashcards)} flashcards generated successfully!")
            else:
                st.warning("No documents were successfully loaded from the uploaded files.")

    st.markdown("---") # Separator

    # --- Internet Knowledge Base Section ---
    st.header("🌐 Generate from Internet Topic")
    internet_topic = st.text_input("Enter a topic (e.g., 'Quantum Computing Basics')")
    num_internet_flashcards = st.slider("Number of flashcards to generate", 3, 15, 5)

    try:

        search_tool = GoogleSearchAPIWrapper()
    except Exception as e:
        st.warning(f"Google Search tool initialization failed: {e}. Ensure you have set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")
        search_tool = None


    if st.button("🔍 Generate Flashcards from Topic"):
        if internet_topic and search_tool:
            with st.spinner(f"Searching the internet for '{internet_topic}' and generating flashcards..."):
                try:
                    # Perform Google Search
                    st.info("Searching the web...")
                    # GoogleSearchAPIWrapper has a .run() method
                    search_results_str = search_tool.run(internet_topic)

                    if not search_results_str or "No good search result found" in search_results_str:
                        st.warning(f"Could not find sufficient search results for '{internet_topic}'. Try a different topic.")
                        st.session_state.flashcards = [] # Clear previous cards
                        st.session_state.current_flashcard_index = 0
                        st.session_state.user_answer = None
                        st.session_state.show_results = False
                        st.session_state.wrong_answers = []
                    else:
                        st.info("Processing search results and generating flashcards with LLM...")
                        # Use the LLM to generate flashcards from search results
                        generated_flashcards = generate_flashcards_with_llm(
                            text_content=search_results_str, 
                            topic=internet_topic,
                            num_flashcards=num_internet_flashcards
                        )

                        if generated_flashcards:
                            st.session_state.flashcards = generated_flashcards
                            st.session_state.current_flashcard_index = 0 
                            st.session_state.user_answer = None
                            st.session_state.show_results = False
                            st.session_state.wrong_answers = [] 

                            st.success(f"Generated {len(generated_flashcards)} flashcards about '{internet_topic}' from internet sources.")
                        else:
                            st.warning(f"Could not generate flashcards from the internet search results for '{internet_topic}'.")

                except Exception as e:
                    st.error(f"An error occurred during internet search or flashcard generation: {e}")
        elif not internet_topic:
            st.warning("Please enter a topic to search.")
        else:
             st.warning("Google Search tool is not available. Cannot perform internet search. Ensure you have set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables and installed langchain-google-community.")


    if st.button("🧹 Reset"):
        st.session_state.flashcards = []
        st.session_state.wrong_answers = []
        st.session_state.current_flashcard_index = 0
        st.session_state.user_answer = None
        st.session_state.show_results = False
        if os.path.exists(st.session_state.db_dir):
            shutil.rmtree(st.session_state.db_dir)
        st.session_state.db_dir = tempfile.mkdtemp()
        st.experimental_rerun()

# --- Flashcard Quiz Interface (Same as before) ---
if st.session_state.flashcards:
    st.header("🎓 Flashcard Quiz")

    if st.session_state.current_flashcard_index < len(st.session_state.flashcards):
        flashcard = st.session_state.flashcards[st.session_state.current_flashcard_index]
        question, correct_answer, choices = flashcard

        st.write(f"**Question:** {question}")

        num_choices = len(choices)
        cols = st.columns(min(num_choices, 2)) 

        for i, choice in enumerate(choices):
            col_index = i % 2
            with cols[col_index]:
                 button_key = f"choice_btn_{st.session_state.current_flashcard_index}_{i}"
                 if st.button(choice, key=button_key, use_container_width=True, disabled=st.session_state.show_results):
                     if not st.session_state.show_results:
                         st.session_state.user_answer = choice
                         st.session_state.show_results = True

                         if choice.lower().strip() != correct_answer.lower().strip():
                             if flashcard not in st.session_state.wrong_answers:
                                 st.session_state.wrong_answers.append(flashcard)
                         st.rerun()

        if st.session_state.show_results:
            st.markdown("---")
            if st.session_state.user_answer.lower().strip() == correct_answer.lower().strip():
                st.success("✅ Correct!")
                if flashcard in st.session_state.wrong_answers:
                     st.session_state.wrong_answers.remove(flashcard)
            else:
                st.error(f"❌ Incorrect! Your answer was: **{st.session_state.user_answer}**")
                st.info(f"The correct answer is: **{correct_answer}**")

            if st.button("Next Question"):
                st.session_state.current_flashcard_index += 1
                st.session_state.user_answer = None
                st.session_state.show_results = False
                st.rerun()

    else:
        st.success("You have completed the quiz!")
        if st.session_state.wrong_answers:
            st.warning(f"You got {len(st.session_state.wrong_answers)} questions wrong. You can click 'Reset' in the sidebar to clear your progress and try again.")

else:
    st.info("Upload and index files or enter a topic to generate flashcards.")