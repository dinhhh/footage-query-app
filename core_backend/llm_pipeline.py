import os
import time
import subprocess
import chromadb
from google import genai
from google.genai import types
from .video_pipeline import extract_video_clip

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
# Replace with your actual Google API key
os.environ["GEMINI_API_KEY"] = "" # input your API key here
client = genai.Client()

# Initialize ChromaDB
DB_PATH = "./cctv_chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
# We use a new collection name to avoid mixing with any old text-based vectors
collection = chroma_client.get_or_create_collection(name="direct_video_vectors")
dimension = 768

# ---------------------------------------------------------
# 4. PHASE 2: Chat & Retrieval (LLM CONTEXT PRUNING)
# ---------------------------------------------------------
def chat_with_raw_video_direct(user_query, raw_video_path, chat_history=None):
    """
    Retrieves top matches, groups them into isolated chronological clusters, 
    extracts clips, and synthesizes an answer using dynamic LLM memory management.
    """
    if chat_history is None:
        chat_history = []

    video_id = os.path.splitext(os.path.basename(raw_video_path))[0]
    print(f"\n=== 🕵️‍♂️ CCTV ASSISTANT QUERY ===\nQuestion: '{user_query}'")

    # --- Step 1: LLM-Driven Conversational Query Rewriting (CQR) ---
    print("🦙 Asking Gemma 4 to resolve context and rewrite the query...")
    
    # Format the entire history into a readable string
    full_history_text = ""
    for idx, turn in enumerate(chat_history):
        full_history_text += f"[Turn {idx+1}] User: {turn['question']} | AI: {turn['answer']}\n"
    
    system_prompt = f"""You are an elite intent parser for a surveillance system.
Your job is to read the user's CURRENT QUERY and the FULL CHAT HISTORY, and output two things separated by a '|' character:
1. RELEVANT HISTORY: A brief summary of ONLY the facts from the history needed to understand the current query. If no history is needed, write 'NONE'.
2. REFINED QUERY: Rewrite the user's current query into a single, fully self-contained natural language sentence. Resolve any pronouns (e.g., replace "he" with "the man in the red shirt") and include necessary context from previous turns so the query makes perfect sense on its own. DO NOT output comma-separated keywords. Write it exactly as a human would ask a complete, standalone question.

FULL CHAT HISTORY:
{full_history_text if full_history_text else "No previous history."}

CURRENT QUERY: {user_query}

Respond strictly in this format: RELEVANT HISTORY | REFINED QUERY"""

    try:
        #local_llm = ChatOllama(model="llama3")
        #ai_message = local_llm.invoke([HumanMessage(content=system_prompt)])

        ai_message = client.models.generate_content(
            model='gemma-4-26b-a4b-it', # Ensure you use the '-it' (Instruction Tuned) version
            contents=system_prompt
        )
        llm_output = ai_message.text.strip()#ai_message.content.strip()


        
        # Robust parsing: Ensure the LLM actually used the '|' delimiter
        if '|' in llm_output:
            extracted_memory, refined_query = [part.strip() for part in llm_output.split('|', 1)]
        else:
            print("⚠️ LLaMA 3 formatting error. Bypassing extraction.")
            extracted_memory = "NONE"
            refined_query = user_query
            
        print(f"🧠 Pruned Memory: {extracted_memory}")
        print(f"✨ Refined Query: {refined_query}")
        
    except Exception as e:
        print(f"⚠️ Local LangChain query rewriting failed: {e}")
        extracted_memory = "NONE"
        refined_query = user_query
    
    # --- Step 2: Embed the Refined Natural Language Query ---
    print("🔢 Embedding refined natural language query...")
    query_embed = client.models.embed_content(
        model='gemini-embedding-2-preview',
        contents=refined_query, # Now using the fully constructed sentence!
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=dimension
        )
    )
    user_vector = query_embed.embeddings[0].values
    
    # --- Step 3: Search ChromaDB ---
    print("🔍 Searching Vector Database for top matches...")
    results = collection.query(
        query_embeddings=[user_vector],
        n_results=5, 
        where={"video_id": video_id}
    )
    
    if not results['metadatas'] or not results['metadatas'][0]:
        final_answer = "I could not find any events matching that description."
        chat_history.append({"question": user_query, "answer": final_answer})
        return final_answer, chat_history
        
    # --- Step 4: Multi-Cluster Logic ---
    chunks = results['metadatas'][0]
    chunks_sorted = sorted(chunks, key=lambda x: x['start_sec'])
    
    clusters = []
    current_cluster = [chunks_sorted[0]]
    
    for chunk in chunks_sorted[1:]:
        last_chunk_end = current_cluster[-1]['end_sec']
        if chunk['start_sec'] - last_chunk_end <= 45.0:
            current_cluster.append(chunk)
        else:
            clusters.append(current_cluster)
            current_cluster = [chunk]
            
    clusters.append(current_cluster)
    print(f"🧩 System identified {len(clusters)} distinct time event(s) across the video.")

    # --- Step 5: Dynamic Trimming & Inline Loading ---
    video_parts = []
    local_clip_paths = []
    
    try:
        for i, cluster in enumerate(clusters):
            cluster_start = min([m['start_sec'] for m in cluster])
            cluster_end = max([m['end_sec'] for m in cluster])
            
            safe_start = max(0.0, cluster_start - 3.0)
            clip_duration = (cluster_end - safe_start) + 3.0 
            
            clip_path = f"temp_cluster_{i}_{int(time.time())}.mp4"
            local_clip_paths.append(clip_path)
            
            print(f"✂️ Extracting Clip {i+1}/{len(clusters)}: {safe_start:.1f}s to {safe_start+clip_duration:.1f}s...")
            extract_video_clip(raw_video_path, safe_start, clip_duration, clip_path)
            
            print(f"📦 Loading Clip {i+1} inline to bypass Cloud Storage...")
            with open(clip_path, "rb") as f:
                video_bytes = f.read()
                
            video_parts.append(
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
            )

        # --- Step 6: Synthesis Continuity (Using Pruned Memory) ---
        print("🧠 Asking Gemini 2.5 Flash to synthesize an answer...")
        
        system_prompt = f"""You are an elite CCTV analysis AI. 
You are being provided with {len(video_parts)} chronologically ordered video clips.
Watch all the clips closely to answer the CURRENT USER QUERY.

INVESTIGATION CONTEXT (Extracted from past turns): 
{extracted_memory}

Use the Investigation Context to resolve any pronouns or track ongoing subjects. 
Return when the event happens, which clip it happens in, and a detailed description."""

        api_contents = video_parts + [system_prompt, f"CURRENT USER QUERY: {user_query}"]

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=api_contents
        )
        final_answer = response.text

    except Exception as e:
        final_answer = f"Error during final visual analysis: {e}"
        
    finally:
        print("🧹 Cleaning up local temporary files...")
        for l_file in local_clip_paths:
            if os.path.exists(l_file):
                os.remove(l_file)
    
    # Save this turn into the session history before returning
    chat_history.append({"question": user_query, "answer": final_answer})
    return final_answer, chat_history
