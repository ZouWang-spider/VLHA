import torch
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont


# 加载预训练的 Mask R-CNN 模型
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# # 加载预训练的 Faster R-CNN 模型
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# 加载预训练的Faster R-CNN模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 设置为评估模式

# 标签映射
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 加载图片
image_path = '7491.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# 进行对象检测
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image_tensor)

# 可视化检测结果
scores = outputs[0]['scores'].numpy()
threshold = 0.6  # 阈值
high_scores_idxs = np.where(scores > threshold)[0]
outputs = [{k: v[high_scores_idxs] for k, v in t.items()} for t in outputs]


def build_scene_graph(outputs):
    instances = outputs[0]
    boxes = instances['boxes']
    labels = instances['labels']
    scores = instances['scores']

    scene_graph = {
        "objects": [],
        "relations": []
    }

    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        if label < len(COCO_INSTANCE_CATEGORY_NAMES):
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            # 将检测到的 "keyboard" 标签替换为 "table"
            if class_name == "keyboard":
                class_name = "dining table"
            scene_graph["objects"].append({
                "id": i,
                "class": class_name,
                "bbox": box.detach().cpu().numpy().tolist(),
                "score": float(score)
            })
        else:
            print(f"Ignore invalid label: {label}")

    # 假设关系：如果两个对象的边界框有重叠，则认为它们有关系
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            box1 = boxes[i].detach().cpu().numpy()
            box2 = boxes[j].detach().cpu().numpy()

            if (box1[0] < box2[2] and box1[2] > box2[0] and
                box1[1] < box2[3] and box1[3] > box2[1]):
                scene_graph["relations"].append({
                    "subject_id": i,
                    "object_id": j,
                    "relation": "overlaps"
                })

    return scene_graph

scene_graph = build_scene_graph(outputs)

# 可用的颜色列表，用于给不同类别的目标分配不同的颜色
colors = ["red", "green", "orange", "magenta", "pink", "purple", "cyan", "lime", "brown", "gray"]

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

color_iter = itertools.cycle(colors)  # 创建无限循环的颜色迭代器

for obj in scene_graph['objects']:
    bbox = obj['bbox']
    class_name = obj['class']
    score = obj['score']

    # 选择颜色，循环使用颜色列表中的颜色
    color = next(color_iter)

    # 绘制边界框
    draw.rectangle(bbox, outline=color, width=2)  # 将线条宽度改为2

    # 计算文本位置和大小
    text = f"{class_name} {score:.2f}"
    text_width, text_height = draw.textsize(text, font=font)
    text_x = bbox[0]
    text_y = bbox[1] - text_height - 1  # 将文本位置上移5个像素，避免与边界框重叠

    # 绘制文本背景小方块
    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)

    # 绘制文本
    draw.text((text_x, text_y), text, fill="white", font=font)  # 文本颜色设为白色


for rel in scene_graph['relations']:
    subject_id = rel['subject_id']
    object_id = rel['object_id']

    if subject_id < len(scene_graph['objects']) and object_id < len(scene_graph['objects']):
        subject_bbox = scene_graph['objects'][subject_id]['bbox']
        object_bbox = scene_graph['objects'][object_id]['bbox']

        subject_center = ((subject_bbox[0] + subject_bbox[2]) / 2, (subject_bbox[1] + subject_bbox[3]) / 2)
        object_center = ((object_bbox[0] + object_bbox[2]) / 2, (object_bbox[1] + object_bbox[3]) / 2)

        draw.line([subject_center, object_center], fill="blue", width=2)

    else:
        print(f"Ignore invalid relation: {subject_id} -> {object_id}")

plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.show()

print("Detected Scene Graph:")
print(scene_graph)





