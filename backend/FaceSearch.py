import os
import sys
import json
import cv2
import numpy as np
import pyodbc
import time
import requests
from io import BytesIO
from dotenv import load_dotenv
from retinaface import RetinaFace
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

requests.packages.urllib3.disable_warnings()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# Load environment variables
load_dotenv()
SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")
API_BASE_URL = os.getenv("API_BASE_URL", "https://localhost:7132")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5")) 

# Initialize FaceNet model
embedder = FaceNet()

# Track processed images
processed_images = set()

def parse_embedding(raw_value):
    """Convert database embedding to numpy array"""
    try:
        if isinstance(raw_value, bytes):
            return np.frombuffer(raw_value, dtype=np.float32)
        elif isinstance(raw_value, str):
            return np.array(json.loads(raw_value), dtype=np.float32)
        return None
    except Exception as e:
        print(f"Error parsing embedding: {e}", file=sys.stderr)
        return None

def save_latest_result(result):
    results_dir = os.path.join(
        r'D:\Anum\WEB\uploader_image\uploader_image',
        "wwwroot",
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    result_path = os.path.join(results_dir, "latest_match.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Results saved to {result_path}")

def process_face_square(img, face, margin_ratio=0.2, target_size=(160, 160)):
    """Crop and resize face to square format"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face["facial_area"]
    
    bw = x2 - x1
    bh = y2 - y1

    margin_x = int(bw * margin_ratio)
    margin_y = int(bh * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    crop_w = x2 - x1
    crop_h = y2 - y1
    
    # Make square
    if crop_w > crop_h:
        diff = crop_w - crop_h
        expand_top = diff // 2
        expand_bottom = diff - expand_top
        if y1 - expand_top >= 0 and y2 + expand_bottom <= h:
            y1 -= expand_top
            y2 += expand_bottom
        else:
            x1 += diff // 2
            x2 -= (diff - diff // 2)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        expand_left = diff // 2
        expand_right = diff - expand_left
        if x1 - expand_left >= 0 and x2 + expand_right <= w:
            x1 -= expand_left
            x2 += expand_right
        else:
            y1 += diff // 2
            y2 -= (diff - diff // 2)

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    cropped = img[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        return None
    
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized

def extract_face_embedding_from_bytes(image_bytes):
    """Extract face embedding from image bytes (no local download)"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image from bytes")
    
    # Detect faces
    faces = RetinaFace.detect_faces(img)
    
    if not isinstance(faces, dict) or len(faces) == 0:
        raise ValueError("No face detected in the uploaded image")
    
    # Use the first detected face
    face_key = list(faces.keys())[0]
    face_data = faces[face_key]
    
    # Crop and process face
    face_crop = process_face_square(img, face_data)
    if face_crop is None:
        raise ValueError("Failed to process detected face")
    
    # Convert to RGB
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    # Generate embedding
    embedding = embedder.embeddings([face_rgb])[0]
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def find_matching_clusters(query_embedding, cluster_threshold=0.55, top_clusters=5):
    """Find top matching person clusters using centroids"""
    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT PersonId, Centroid, Count
            FROM dbo.PersonPrototypes
            WHERE Centroid IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            return []
        
        cluster_matches = []
        for person_id, centroid_blob, count in rows:
            centroid = parse_embedding(centroid_blob)
            if centroid is None:
                continue
            
            centroid = centroid / np.linalg.norm(centroid)
            similarity = float(cosine_similarity([query_embedding], [centroid])[0][0])
            
            if similarity >= cluster_threshold:
                cluster_matches.append({
                    "personId": str(person_id),
                    "similarity": similarity,
                    "faceCount": count
                })
        
        cluster_matches.sort(key=lambda x: x["similarity"], reverse=True)
        return cluster_matches[:top_clusters]
        
    except Exception as e:
        print(f"Database error in cluster matching: {e}", file=sys.stderr)
        raise

def find_faces_in_clusters(query_embedding, person_ids, face_threshold=0.6):
    """Find matching faces within shortlisted person clusters"""
    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?' for _ in person_ids])
        
        query = f"""
            SELECT 
                f.Id AS FaceId,
                f.MediaFileId,
                f.FrameNumber,
                f.Embedding,
                f.PersonId,
                mf.FileName,
                mf.FilePath
            FROM dbo.Faces f
            INNER JOIN dbo.MediaFiles mf ON f.MediaFileId = mf.Id
            WHERE f.PersonId IN ({placeholders})
              AND f.Embedding IS NOT NULL
        """
        
        cursor.execute(query, person_ids)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            return []
        
        matches = []
        for row in rows:
            face_id, media_id, frame_num, emb_blob, person_id, filename, filepath = row
            
            db_embedding = parse_embedding(emb_blob)
            if db_embedding is None:
                continue
            
            db_embedding = db_embedding / np.linalg.norm(db_embedding)
            similarity = float(cosine_similarity([query_embedding], [db_embedding])[0][0])
            
            if similarity >= face_threshold:
                matches.append({
                    "faceId": str(face_id),
                    "mediaFileId": str(media_id),
                    "frameNumber": int(frame_num) if frame_num else 0,
                    "fileName": filename,
                    "filePath": filepath,
                    "similarity": round(similarity, 4),
                    "personId": str(person_id) if person_id else None
                })
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches
        
    except Exception as e:
        print(f"Database error in face matching: {e}", file=sys.stderr)
        raise

def fetch_image_list():
    """Fetch list of uploaded images from API"""
    try:
        list_url = f"{API_BASE_URL}/api/list-images"
        response = requests.get(list_url, verify=False, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching image list: {e}", file=sys.stderr)
        return []

def fetch_image_bytes(image_url):
    """Fetch image bytes from URL without downloading to disk"""
    try:
        response = requests.get(image_url, verify=False, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error fetching image from {image_url}: {e}", file=sys.stderr)
        return None

def process_image_from_url(image_url):
    """Process image directly from URL"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing new image: {image_url}")
        print(f"{'='*60}")
        
        # Fetch image bytes
        image_bytes = fetch_image_bytes(image_url)
        if image_bytes is None:
            raise ValueError("Failed to fetch image")
        
        # Extract embedding from bytes (no local download)
        query_embedding = extract_face_embedding_from_bytes(image_bytes)
        
        # Find matching clusters
        matching_clusters = find_matching_clusters(
            query_embedding, 
            cluster_threshold=0.55, 
            top_clusters=5
        )
        
        if not matching_clusters:
            result = {
                "success": True,
                "imageUrl": image_url,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "matchCount": 0,
                "matches": [],
                "message": "No matching person clusters found"
            }
            print(json.dumps(result, indent=2))
            save_latest_result(result)

            return
        
        # Find faces within clusters
        person_ids = [cluster["personId"] for cluster in matching_clusters]
        matches = find_faces_in_clusters(
            query_embedding, 
            person_ids, 
            face_threshold=0.6
        )
        
        # Prepare result
        result = {
            "success": True,
            "imageUrl": image_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "matchCount": len(matches),
            "clusterCount": len(matching_clusters),
            "clusters": matching_clusters,
            "matches": matches
        }
        
        print(json.dumps(result, indent=2))
        save_latest_result(result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "imageUrl": image_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }
        print(json.dumps(error_result, indent=2))
        save_latest_result(result)

def monitor_api():
    """Continuously poll API for new images"""
    global processed_images
    
    print(f"{'='*60}")
    print(f"Face Detection API Monitor Started")
    print(f"{'='*60}")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Poll Interval: {POLL_INTERVAL} seconds")
    print(f"Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Fetch list of uploaded images
            image_urls = fetch_image_list()
            
            if image_urls:
                # Process new images only
                new_images = [url for url in image_urls if url not in processed_images]
                
                if new_images:
                    print(f"Found {len(new_images)} new image(s)")
                    
                    for image_url in new_images:
                        process_image_from_url(image_url)
                        processed_images.add(image_url)
            
            # Wait before next poll
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        print("Monitor stopped.")

def single_image_mode(image_url):
    """Process a single image from URL"""
    try:
        image_bytes = fetch_image_bytes(image_url)
        if image_bytes is None:
            raise ValueError("Failed to fetch image")
        
        query_embedding = extract_face_embedding_from_bytes(image_bytes)
        
        matching_clusters = find_matching_clusters(
            query_embedding, 
            cluster_threshold=0.55, 
            top_clusters=5
        )
        
        if not matching_clusters:
            result = {
                "success": True,
                "matchCount": 0,
                "matches": [],
                "message": "No matching person clusters found"
            }
            print(json.dumps(result))
            save_latest_result(result)

            sys.exit(0)
        
        person_ids = [cluster["personId"] for cluster in matching_clusters]
        matches = find_faces_in_clusters(
            query_embedding, 
            person_ids, 
            face_threshold=0.6
        )
        
        result = {
            "success": True,
            "matchCount": len(matches),
            "clusterCount": len(matching_clusters),
            "clusters": matching_clusters,
            "matches": matches
        }
        
        print(json.dumps(result))
        save_latest_result(result)

        sys.exit(0)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        save_latest_result(error_result)

        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        # No arguments - start monitor mode
        monitor_api()
    elif sys.argv[1] == "--monitor":
        # Explicit monitor mode
        monitor_api()
    else:
        # Single image URL mode
        image_url = sys.argv[1]
        single_image_mode(image_url)

if __name__ == "__main__":
    main()