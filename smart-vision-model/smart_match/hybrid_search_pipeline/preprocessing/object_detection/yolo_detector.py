from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def detect_objects_with_yolov8(image_path, model_name='yolov8n.pt', conf=0.5):
    """
    YOLOv8을 이용한 객체 탐지 및 바운딩 박스 시각화

    Args:
        image_path (str): 입력 이미지 경로
        model_name (str): yolov8n.pt / yolov8s.pt / yolov8m.pt 등
        conf (float): confidence threshold (기본값 0.5)

    Returns:
        result_list (List[dict]): 객체 탐지 결과 (좌표, 클래스명)
    """
    # 모델 로드
    model = YOLO(model_name)

    # 이미지 로드 및 예측
    results = model(image_path, conf=conf)

    # 결과 그리기
    res_plotted = results[0].plot()  # numpy array(BGR)

    # 결과 리스트 정리
    result_list = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
        confidence = float(box.conf[0])

        result_list.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': xyxy
        })

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"YOLOv8 Detection ({model_name})")
    plt.savefig(".output.jpg")
    #plt.show()

    return result_list

if __name__ == "__main__":
    image_path = '../../../../data/binary_label_classifier/Chiller & Scrubber/Accretech SCU-500R/L_Accretech-SCU-500R-241210194410803-004.jpg'
    detections = detect_objects_with_yolov8(image_path, model_name='yolov8n.pt', conf=0.1)

    for det in detections:
        print(f"{det['class']} ({det['confidence']:.2f}) -> {det['bbox']}")
