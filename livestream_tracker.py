# --- 1. KONFIGURACJA ---
# Ścieżka do modelu
model_path = r"C:\Users\szymo\Desktop\DTU\3rd_semester\Individual_project_demo\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)

CAMERA_SOURCE = "https://stream.sob.m-dn.net/live/sb1/vKVhWPO2ysiYNGrNfA+Krw1/stream.m3u8?plain=true"
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.0
WATER_REGION_PERCENT = 0.6
PROCESS_EVERY_N_FRAMES = 1 
SHOW_WATER_LINE = False

# --- KONFIGURACJA ZAPISU AUTOMATYCZNEGO ---
SAVE_DETECTIONS = True
OUTPUT_FOLDER = r"C:\Users\szymo\Desktop\DTU\3rd_semester\Individual_project_demo\live_detections"
SAVE_COOLDOWN = 2.0  # Ile sekund czekać przed kolejnym zapisem tego samego obiektu

if SAVE_DETECTIONS:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. KLASA DO OBSŁUGI KAMERY BEZ LAGÓW ---
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.ret, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.ret, self.frame) = self.stream.read()
            time.sleep(0.005)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- 3. FUNKCJE POMOCNICZE ---
def create_water_mask(frame_shape):
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    water_start_y = int(height * (1 - WATER_REGION_PERCENT))
    mask[water_start_y:, :] = 255
    return mask, water_start_y

def filter_detections_in_roi(boxes, mask):
    filtered_indices = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1]:
            if mask[center_y, center_x] > 0:
                filtered_indices.append(idx)
    return filtered_indices

def draw_water_line(frame, water_start_y):
    cv2.line(frame, (0, water_start_y), (frame.shape[1], water_start_y), (0, 255, 255), 2)
    cv2.putText(frame, "Water Region", (10, water_start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# --- 4. INTERFEJS ---
start_button = widgets.Button(description="Start Auto-Detect", button_style='success')
stop_button = widgets.Button(description="Stop Camera", button_style='danger')
status_label = widgets.Label(value="Ready")
image_widget = widgets.Image(format='jpeg', width=800, height=600)

display(widgets.HBox([start_button, stop_button]))
display(status_label)
display(image_widget)

running = False
video_stream = None

def start_camera(b):
    global running
    if not running:
        running = True
        status_label.value = "Starting..."
        Thread(target=main_processing_loop, daemon=True).start()

def stop_camera(b):
    global running
    running = False
    status_label.value = "Stopping..."

start_button.on_click(start_camera)
stop_button.on_click(stop_camera)

# --- 5. GŁÓWNA PĘTLA Z AUTO-ZAPISEM ---
def main_processing_loop():
    global running, video_stream
    
    video_stream = CameraStream(src=CAMERA_SOURCE).start()
    time.sleep(1.0) # Rozgrzewka kamery
    
    first_frame = video_stream.read()
    if first_frame is None:
        status_label.value = "Error: Camera returned None"
        running = False
        return

    water_mask, water_line_y = create_water_mask(first_frame.shape)
    
    frame_counter = 0
    detection_total_count = 0
    cached_boxes = []
    
    # Zmienna do kontroli czasu zapisu
    last_save_time = 0
    
    while running:
        frame = video_stream.read()
        if frame is None:
            continue

        frame_counter += 1
        annotated_frame = frame.copy()
        
        # --- DETEKCJA ---
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            results = model.predict(source=frame, save=False, show=False, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            
            cached_boxes = []
            if len(results[0].boxes) > 0:
                valid_indices = filter_detections_in_roi(results[0].boxes, water_mask)
                if valid_indices:
                    detection_total_count += 1 # To tylko licznik ogólny
                    for idx in valid_indices:
                        box = results[0].boxes[idx]
                        coords = tuple(map(int, box.xyxy[0].cpu().numpy()))
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        cached_boxes.append((coords, conf, cls))

        # --- RYSOWANIE ---
        detections_now = len(cached_boxes) # Ile obiektów widzimy TERAZ na ekranie
        
        for (coords, conf, cls) in cached_boxes:
            x1, y1, x2, y2 = coords
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if SHOW_WATER_LINE:
            draw_water_line(annotated_frame, water_line_y)

        # Info overlay
        info_text = f"Auto-Save Mode | Detections: {detections_now}"
        # Jeśli trwa cooldown, wyświetl informację
        if time.time() - last_save_time < SAVE_COOLDOWN:
            color = (0, 0, 255) # Czerwony (czekanie)
        else:
            color = (0, 255, 0) # Zielony (gotowy do zapisu)
            
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- AUTO SAVE LOGIC ---
        # Jeśli coś wykryto I minął czas cooldownu
        current_time = time.time()
        if detections_now > 0 and (current_time - last_save_time > SAVE_COOLDOWN):
            if SAVE_DETECTIONS:
                ts_filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # %f dodaje milisekundy dla unikalności
                save_path = os.path.join(OUTPUT_FOLDER, f"auto_detect_{ts_filename}.jpg")
                cv2.imwrite(save_path, annotated_frame)
                
                last_save_time = current_time # Reset licznika czasu
                status_label.value = f"AUTO-SAVED: {save_path}"

        # Wyświetlanie
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        image_widget.value = buffer.tobytes()
        
        time.sleep(0.001)

    video_stream.stop()
    status_label.value = "Stopped"

print("=== YOLO Auto-Save Mode ===")
print(f"Zdjęcia będą zapisywane automatycznie co {SAVE_COOLDOWN}s, gdy wykryto obiekt.")