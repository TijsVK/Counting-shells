import requests
from PIL import Image, ImageFont, ImageDraw
import torch
import cv2
import torchvision
import numpy as np

from transformers import OwlViTProcessor, OwlViTForObjectDetection, AutoProcessor

USE_TEXT = False
DO_NMS = True

processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

path = "../../../Data/shells/IMG20230217104337.jpg"
impath = "../../../Data/Fixed_support/base/"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(path)
texts = [["baltic_tellin", "cockle", "cut_through_shell", "oyster", "wedge", "mussel"]]
if USE_TEXT:
    inputs = processor(text=texts, images=image, return_tensors="pt")
else:
    query_paths = [impath + base + ".jpg" for base in texts[0]]
    query_imgs = [Image.open(impath) for impath in query_paths]
    print(query_imgs, image)
    inputs = processor(query_images=query_imgs, images=image, return_tensors="pt")

for key, val in inputs.items():
    print(f"{key}: {val.shape}")

if USE_TEXT:
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    
else:
    outputs = model.image_guided_detection(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()
    outputs.pred_boxes = outputs.target_pred_boxes.cpu()
    target_sizes = torch.Tensor([image.size[::-1]] * len(query_imgs))

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
# Convert outputs (bounding boxes and class logits) to COCO API
score_threshold = 0.1
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold = score_threshold)

# Retrieve predictions for the first image for the corresponding text queries
text = texts[0]
boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
for i in range(1, len(text)):
    im_boxes, im_scores, im_labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    boxes = torch.cat((boxes, im_boxes), dim=0)
    scores = torch.cat((scores, im_scores), dim=0)
    # add offset i to labels
    im_labels = im_labels + i
    labels = torch.cat((labels, im_labels), dim=0)


# non-maximum suppression with torchvision
if DO_NMS:
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.2)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        
# draw bounding boxes on image
draw = ImageDraw.Draw(image)
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        draw.rectangle(box, outline="red", width=10)
        #add label and score
        draw.text((box[0]+10, box[1]+10), f"{text[label]} {round(score.item(), 3)}", fill="red", font=ImageFont.truetype("arial", 120))

image.save(path.split("/")[-1].split(".")[0]+"_owlvit.jpg")