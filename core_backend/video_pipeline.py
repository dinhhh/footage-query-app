import os
import time
import subprocess
import chromadb
from google import genai
from google.genai import types

os.environ["GEMINI_API_KEY"] = "" # input your API key here
client = genai.Client()

# Initialize ChromaDB
DB_PATH = "./cctv_chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
# We use a new collection name to avoid mixing with any old text-based vectors
collection = chroma_client.get_or_create_collection(name="direct_video_vectors")
dimension = 768

# ---------------------------------------------------------
# 2. FFmpeg Video Utilities
# ---------------------------------------------------------
def get_video_duration(video_path):
    """Uses FFprobe to get the exact duration of the raw video."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    return float(subprocess.check_output(cmd).decode('utf-8').strip())

def extract_video_clip(input_path, start_sec, duration, output_path):
    """Losslessly slices a specific time window from the raw video."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", input_path, 
        "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "make_zero", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path if os.path.exists(output_path) else None

# ---------------------------------------------------------
# 3. PHASE 1: Direct Multimodal Ingestion (Run Once)
# ---------------------------------------------------------

def ingest_raw_video_direct(video_path, chunk_duration=15.0):
    """
    Directly embeds raw video chunks into ChromaDB using INLINE DATA,
    bypassing the Google Cloud File API entirely.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    duration = get_video_duration(video_path)
    
    print(f"\n=== 🎬 STARTING DIRECT MULTIMODAL INGESTION: {video_id} ({duration:.1f}s) ===")
    
    current_sec = 0.0
    chunk_idx = 0
    
    while current_sec < duration:
        end_sec = min(current_sec + chunk_duration, duration)
        clip_path = f"temp_ingest_{chunk_idx}.mp4"
        
        # 1. Extract 15-second Chunk
        extract_video_clip(video_path, current_sec, chunk_duration, clip_path)
        print(f"\n⚙️ Processing Chunk {chunk_idx} [{current_sec:.1f}s - {end_sec:.1f}s]...")
        
        try:
            # 2. READ VIDEO AS RAW BYTES (Bypasses the buggy File API)
            with open(clip_path, "rb") as f:
                video_bytes = f.read()
            
            # 3. DIRECT EMBEDDING (Inline Bytes -> Math)
            embed_response = client.models.embed_content(
                model='gemini-embedding-2-preview',
                # Pass the raw bytes directly to the model
                contents=types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dimension
                )
            )
            vector = embed_response.embeddings[0].values
            
            # 4. Save Vector to ChromaDB 
            collection.add(
                embeddings=[vector],
                documents=[""], # No text!
                metadatas=[{"video_id": video_id, "start_sec": current_sec, "end_sec": end_sec}],
                ids=[f"{video_id}_chunk_{chunk_idx}"]
            )
            print("   ✅ Video chunk embedded and saved.")
            
        except Exception as e:
            print(f"   ❌ Error processing chunk {chunk_idx}: {e}")
            
        # Clean up local file
        if os.path.exists(clip_path):
            os.remove(clip_path)
            
        current_sec += chunk_duration
        chunk_idx += 1
        
    print(f"=== 🎉 DIRECT INGESTION COMPLETE! '{video_id}' is now searchable. ===")
