import spacy
import fitz  # PyMuPDF for PDF processing

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# 📌 1️⃣ Read PDF and extract text (optimized version)
def extract_text_from_pdf(pdf_path):
    doc_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text")
                if text.strip():  # Only add non-empty pages
                    doc_text += text + "\n"
                else:
                    print(f"Warning: Page {page.number} might be empty or an image.")
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return doc_text

# 📌 Test optimized PDF extraction
pdf_path = "2024-Annual-Report.pdf"
full_text = extract_text_from_pdf(pdf_path)

# Preview the first 500 characters to ensure text extraction works correctly
print(full_text[:500])

# 📌 2️⃣ Extract only government regulation-related paragraphs (optimized version)
def extract_regulation_sections(text):
    regulation_keywords = ["regulation", "law", "policy", "compliance", "government",
                           "FCC", "SEC", "GDPR", "COPPA", "legal", "data protection", "privacy"]

    relevant_paragraphs = []
    paragraphs = text.split("\n\n")  # Split text into paragraphs

    for para in paragraphs:
        doc = nlp(para)  # Use spaCy for NLP analysis
        if any(keyword.lower() in para.lower() for keyword in regulation_keywords):
            # Check if regulatory organizations or legal terms are mentioned
            if any(ent.label_ in ["ORG", "LAW", "GPE"] for ent in doc.ents):
                relevant_paragraphs.append(para)

    return "\n\n".join(relevant_paragraphs)

# 📌 Extract regulation-related content
regulation_text = extract_regulation_sections(full_text)

# 📌 Save regulation content to a file
regulation_file = "regulation_sections1.txt"
with open(regulation_file, "w", encoding="utf-8") as f:
    f.write("Extracted Regulation-Related Sections:\n")
    f.write(regulation_text)

print(f"Regulation-related content saved to: {regulation_file}")

from collections import Counter

# 📌 3️⃣ Run NER and save identified entities (optimized version)
def analyze_text_with_ner(text, output_file):
    doc = nlp(text)
    entity_counts = Counter()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Named Entity Recognition (NER) Results:\n")

        for ent in doc.ents:
            if ent.label_ in ["ORG", "LAW", "GPE"]:
                entity_counts[ent.text] += 1
                f.write(f"Entity: {ent.text}, Label: {ent.label_}\n")

        f.write("\nTop Mentioned Regulatory Entities:\n")
        for entity, count in entity_counts.most_common(10):
            f.write(f"{entity}: {count} times\n")

# 📌 Run NER and save results
ner_file = "ner_results1.txt"
analyze_text_with_ner(regulation_text, ner_file)

print(f"NER results saved to: {ner_file}")

# Read regulation-related content directly from TXT
regulation_file = "regulation_sections1.txt"
with open(regulation_file, "r", encoding="utf-8") as f:
    regulation_text = f.read()

print("Regulation text successfully loaded from file!")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_openai import OpenAIEmbeddings

# Set API Key in the environment variables
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Ensure API Key exists
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API Key is not set. Please check environment variables!")
    exit()

try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    print("OpenAI Embeddings initialized successfully!")
except Exception as e:
    print(f"OpenAI connection failed: {e}")
    exit()

# Step 1: Split regulation text into chunks (1000 characters per chunk)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = splitter.split_text(regulation_text)

# Step 2: Generate text embeddings
embeddings = OpenAIEmbeddings()

# Step 3: Store in Chroma vector database
vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings)

print("Regulation text has been successfully stored in the vector database!")

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

# Step 1: Load OpenAI LLM
llm = OpenAI(temperature=0)

# Step 2: Create retriever
retriever = vectorstore.as_retriever()

# Step 3: Set up conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 4: Create QA system
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

print("AI Question-Answering System successfully initialized!")

# Step 5: Run multiple test questions
questions = [
    "What are the main regulations mentioned in the document?",
    "How does Disney ensure compliance with data protection laws?",
    "What legal challenges has Disney faced in recent years?",
    "Does Disney comply with GDPR and CCPA regulations?",
    "How does Disney handle customer data security?",
]

# Step 6: Execute and print answers
for i, question in enumerate(questions, 1):
    result = qa_chain.invoke({"question": question})  # Use invoke() instead of run()
    print(f"\nQuestion {i}: {question}")
    print(f"AI Answer: {result}\n")

