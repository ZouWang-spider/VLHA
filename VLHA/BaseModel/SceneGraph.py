import torch
import cv2
import itertools
import numpy as np
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont

# Mask R-CNN model Trained
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# Faster R-CNN model Trained
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

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

#traget detection
def Image_traget_detection(image, threshold):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
    # 可视化检测结果
    scores = outputs[0]['scores'].numpy()
    high_scores_idxs = np.where(scores > threshold)[0]
    outputs = [{k: v[high_scores_idxs] for k, v in t.items()} for t in outputs]
    return outputs

#Bulid Viausl Scene Graph
def build_scene_graph(outputs, threshold):
    instances = outputs[0]
    boxes = instances['boxes']
    labels = instances['labels']
    scores = instances['scores']

    scene_graph = {
        "objects": [],
        "relations": []
    }

    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        if score >= threshold:  # 只保留大于阈值的目标
            if label < len(COCO_INSTANCE_CATEGORY_NAMES):
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                # 将检测到的 "keyboard" 标签替换为 "dining table"
                if class_name == "keyboard":
                    class_name = "dining table"
                if class_name == "snowboard":
                    class_name = "tie"
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

#show scene graph
def scene_draw(image, scene_graph):
    # 可用的颜色列表，用于给不同类别的目标分配不同的颜色
    colors = ["red", "green", "orange", "magenta", "pink", "purple", "cyan", "lime", "brown", "gray"]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    color_iter = itertools.cycle(colors)  # 创建无限循环的颜色迭代器

    #target box
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

    #Traget relation
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
    #show
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Using ResNet model Extraction target regions as node features
resnet_model = resnet50(pretrained=True)
resnet_model.eval()
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))  # 去掉最后的分类层

def extract_features(image, scene_graph):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    for obj in scene_graph['objects']:
        bbox = obj['bbox']
        cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        input_tensor = transform(cropped_img).unsqueeze(0)
        with torch.no_grad():
            feature = resnet_model(input_tensor)
        features.append(feature.squeeze().cpu().numpy())
    # 将特征列表转化为张量
    features_tensor = torch.tensor(features)
    return features_tensor


#Bulid Visual Graph Structure
def build_graph_structure(scene_graph, features):
    # 创建节点列表
    nodes = []
    for i, obj in enumerate(scene_graph['objects']):
        nodes.append({
            'id': i,  # 使用 i 作为节点的连续索引
            'feature': features[i],
            'class': obj['class']
        })

    # 创建边列表，并确保边的索引不超过节点列表的长度
    edges = []
    for rel in scene_graph['relations']:
        source_id = rel['subject_id']
        target_id = rel['object_id']

        # 只添加有效的边，即索引在节点列表范围内的边
        if source_id < len(nodes) and target_id < len(nodes):
            edges.append({
                'source': source_id,
                'target': target_id,
                'relation': rel['relation']
            })

    # 将边的关系转化为张量
    edge_tensor = torch.tensor([(edge['source'], edge['target']) for edge in edges]).t()

    return nodes, edge_tensor


# #model fit train
# image_path = 'E:/PythonProject2/VLHA/Datasets/7491.jpg'
# threshold = 0.7  # 阈值
# image = Image.open(image_path)
#
# outputs = Image_traget_detection(image, threshold)
# scene_graph = build_scene_graph(outputs, threshold)
# scene_draw(image, scene_graph)
# node_feature = extract_features(image, scene_graph)   # n,2048
# print(node_feature.shape)
# nodes, edge_tensor = build_graph_structure(scene_graph, node_feature)  #tensor([[0, 1, 1, 2], [3, 2, 3, 3]])
# # print(nodes)
# print("2",edge_tensor)
#
# print("Detected Scene Graph:")
# print(scene_graph)





