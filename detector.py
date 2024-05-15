from ultralytics import YOLO

def pred(img):  
  model = YOLO('yolov8n.pt')
  results = model.predict(img)
  result = results[0]
  box = result.boxes[0]
  return print(result.names[box.cls[0].item()])
