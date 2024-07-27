from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict



def get_image_tensor(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)   #对图像进行预处理转化为torch类型
    return image



def load_dataset(data_file,image_path):
    with open(data_file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    # # 处理句子中包含多词方面词, 使用 defaultdict 创建一个以 sentence 为键，值为包含该句子信息的列表的字典
    # sentence_dict = defaultdict(lambda: {"sentence": "", "aspect_terms": [], "sentiments": [], "image_path": ""})

    num_samples = len(lines) // 4    #数据集数量
    samples = []

    for i in range(num_samples):
        sentence = lines[i * 4].strip()
        aspect_term = lines[i * 4 + 1].strip()
        sentiment = int(lines[i * 4 + 2].strip())
        image_filename = lines[i * 4 + 3].strip()

        # Replace $T$ with aspect_term in the sentence
        sentence = sentence.replace('$T$', aspect_term)

        # 将信息添加到字典中
        # sentence_dict[sentence]["sentence"] = sentence
        # sentence_dict[sentence]["aspect_terms"].append(aspect_term)
        # sentence_dict[sentence]["sentiments"].append(sentiment)
        # # sentence_dict[sentence]["image_filename"] = image_filename

        # 图像路径拼接
        full_image_path = os.path.join(image_path, image_filename)

        # 读取并预处理图像
        image_tensor = get_image_tensor(full_image_path, transform=None)
        # sentence_dict[sentence]["image_path"] = full_image_path

        sample = {
            "sentence": sentence,
            "aspect_term": aspect_term,
            "sentiment": sentiment,
            "image_filename": image_filename,
            "image_path": full_image_path

        }



        # # 将字典的值转换为列表
        # samples = list(sentence_dict.values())
        samples.append(sample)


    return samples


# data_path = "E:/PythonProject2/VisionLanguageMABSA/Dataset/twitter2015/train.txt"
# image_path = "E:/PythonProject2/VisionLanguageMABSA/Dataset/twitter2015_images"
# samples = load_dataset(data_path,image_path)
#
# for sample in samples:
#     print(sample)
#     print(sample['image_path'])
#     print(sample['aspect_term'])
#     print(sample['sentiment'])

