import carla
import cv2
import numpy as np
import random
import os
import time
from collections import deque
import torch
import torchvision.transforms as T
import torchvision
from ultralytics import YOLO

image_queue = deque(maxlen=50)

def camera_callback(image):
    image_queue.append(image)


def obtain_bounding_boxes(result, model):

    target_classes = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light"
    }

    for box in result.boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Confidence
        confidence = box.conf[0].item()

        # Class ID
        class_id = int(box.cls[0].item())

        # Class name
        class_name = model.names[class_id]

        if class_name not in target_classes:
            continue

        print(
            f"{class_name}: "
            f"{confidence:.2f} "
            f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

    return 0


def main():

    SAVE_IMAGES = False
    OUTPUT_DIR = "vo_frames"

    if SAVE_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()


    # Spawn vehicle
    vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    spawn_points = world.get_map().get_spawn_points()

    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))

    print("Vehicle spawned")

    # Enable autopilot so the car moves
    traffic_manager = client.get_trafficmanager()
    vehicle.set_autopilot(True, traffic_manager.get_port())

    # RGB Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '640')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    camera.listen(camera_callback)

    print("Camera started")


    # =====================================================
    # Main Loop
    # =====================================================
    model = YOLO("yolo26n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    frame_count = 0

    try:

        while True:

            if len(image_queue) < 2:
                continue

            image = image_queue.popleft()

            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))

            rgb = array[:, :, :3] #(600,800,3)

            rgb = rgb.copy()

            with torch.no_grad():
                output = model.track(rgb, persist=True, conf=0.6, tracker="bytetrack.yaml")

            results = output[0]

            
            if results.boxes.id is None:
                cv2.imshow("CARLA YOLO", rgb)
                cv2.waitKey(1)
                continue
            
            boxes = results.boxes.xywh.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()
            conf_scores = results.boxes.conf.cpu().tolist()
            boxes_class_ids = results.boxes.cls.int().cpu().tolist()
            boxes_classnames = [results.names[class_id] for class_id in boxes_class_ids]

            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            for box, track_id, conf, class_name_new in zip(boxes, track_ids, conf_scores, boxes_classnames):
                x_center = box[0].item()
                y_center = box[1].item()
                w        = box[2].item()
                h        = box[3].item()

                x1 = int((x_center - (w / 2)))
                y1 = int((y_center - (h / 2)))
                x2 = int((x_center + (w / 2)))
                y2 = int((y_center + (h / 2)))

                

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label = (
                            f"{class_name_new} "
                            f"ID:{track_id} "
                            f"{conf:.2f}"
                        )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            cv2.imshow("CARLA YOLO Tracking", frame)

            frame_count += 1

            # Quit with Q
            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:

        print("Cleaning up...")

        camera.stop()

        camera.destroy()
        vehicle.destroy()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()