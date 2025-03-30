import cv2
import asyncio
import time
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
import time

# Global variables for sharing data between async functions
frame_to_process = None
processed_frame = None
is_frame_ready = False
is_processing_done = True
face_id = 0
first = True
last = 0
last_area_time = 0
area_id = 0

async def capture_frames(cap, target_fps=60):

    global frame_to_process, is_frame_ready, is_processing_done, last_area_time, area_id
    
    frame_delay = 1 / target_fps
    prev_frame_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed = current_time - prev_frame_time
        
        if elapsed >= frame_delay:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            if last_area_time + 60 < current_time:
                last_area_time = current_time
                cv2.imwrite(f'area/output_image_{area_id}.jpg', frame)
                print(f"Saved area image as output_image_{area_id}.jpg")
                area_id += 1

            # Wait until the previous frame has been processed
            if is_processing_done:
                frame_to_process = frame.copy()
                is_frame_ready = True
                is_processing_done = False
                prev_frame_time = current_time
        
        # Small sleep
        await asyncio.sleep(0.01)

def detect_faces(frame):
    """Detect faces in a frame and draw red rectangles around them only if faces are detected"""
    global face_id, first, last
    result_frame = frame.copy()
    face_detected = False
    
    try:
        faces = DeepFace.extract_faces(img_path=frame, 
                                      detector_backend='opencv',
                                      enforce_detection=True)
        
        # Only draw rectangles if faces were actually detected
        if faces and len(faces) > 0:
            face_detected = True
            for face in faces:
                facial_area = face['facial_area']
                #print(f"Detected face at: {facial_area}")
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                
                left = x - 10
                top = y - 70
                right = x + w + 5
                bottom = y + h + 20

                cv2.rectangle(result_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                
                try:

                    if first == True:
                        face_id = 0
                        first = False
                        last = time.time()
                        crop_face = frame[top:bottom, left:right]
                        resized_crop_face = cv2.resize(crop_face, (128, 128))
                        cv2.imwrite(f'face/output_image_{face_id}.jpg', resized_crop_face)
                        
                        print(f"Saved cropped face image as output_image_{face_id}.jpg")
                        face_id += 1
                    else:
                        if last + 3 < time.time():
                            last = time.time()
                            crop_face = frame[top:bottom, left:right]
                            resized_crop_face = cv2.resize(crop_face, (128, 128))
                            cv2.imwrite(f'D:/hackpsuS25/hackpsu_S25/face/output_image_{face_id}.jpg', resized_crop_face)
                            print(f"Saved cropped face image as output_image_{face_id}.jpg")
                            face_id += 1
                except Exception as e:
                    print(f"Error saving cropped face image: {e}")
                    
                
    
    except Exception as e:
        print(f"THERE IS NO FACE: {e}")
    
    # Return the original frame if no faces were detected (no red boxes)
    return result_frame, face_detected

async def process_frames(executor):
    global frame_to_process, processed_frame, is_frame_ready, is_processing_done
    
    while True:
        if is_frame_ready:
            # Run face detection in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            processed_frame, _ = await loop.run_in_executor(
                executor, detect_faces, frame_to_process.copy())
            
            is_frame_ready = False
            is_processing_done = True
        
        # Small sleep
        await asyncio.sleep(0.01)

async def display_frames(window_name="Face Detection"):
    global processed_frame, is_processing_done
    
    while True:
        if processed_frame is not None:
            cv2.imshow(window_name, processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting application...")
                return
        
        # Small sleep
        await asyncio.sleep(0.01)

async def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create a thread pool for the face detection
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Create tasks
        capture_task = asyncio.create_task(capture_frames(cap))
        process_task = asyncio.create_task(process_frames(executor))
        display_task = asyncio.create_task(display_frames())
        
        # Wait for display task to complete (when user presses 'q')
        await display_task
        
        # Cancel other tasks
        capture_task.cancel()
        process_task.cancel()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


asyncio.run(main())  