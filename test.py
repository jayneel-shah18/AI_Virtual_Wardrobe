import time
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
import cv2

SIZE = 320
NC = 14

def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)
    return label_batch

def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], NC))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)
    return input_label

def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int32)
    M_f = torch.FloatTensor(M_f)
    masked_img = img * (1 - mask)
    M_c = (1 - mask) * M_f
    M_c = M_c + torch.zeros(img.shape)  # broadcasting
    return masked_img, M_c, M_f

def compose(label, mask, color_mask, edge, color, noise):
    masked_label = label * (1 - mask)
    masked_edge = mask * edge
    masked_color_strokes = mask * (1 - color_mask) * color
    masked_noise = mask * noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise

def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == 11).astype(int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == 13).astype(int))
    noise = torch.FloatTensor((old_label.cpu().numpy() == 7).astype(int))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label

def main():
    os.makedirs('sample', exist_ok=True)
    opt = TestOptions().parse()
    opt.checkpoints_dir = "AI_Virtual_Wardrobe/checkpoints"
    opt.name = "label2city"

    # Force CPU mode if CUDA not available
    opt.gpu_ids = []
    torch.cuda.is_available = lambda: False

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    model = create_model(opt)

    for i, data in enumerate(dataset):
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(int))
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(int))
        img_fore = data['image'] * mask_fore
        all_clothes_label = changearm(data['label'])

        fake_image, warped_cloth, refined_cloth = model(
            Variable(data['label']),
            Variable(data['edge']),
            Variable(img_fore),
            Variable(mask_clothes),
            Variable(data['color']),
            Variable(all_clothes_label),
            Variable(data['image']),
            Variable(data['pose']),
            Variable(data['image']),
            Variable(mask_fore)
        )

        output_dir = os.path.join(opt.results_dir, opt.phase)
        fake_image_dir = os.path.join(output_dir, 'try-on')
        warped_cloth_dir = os.path.join(output_dir, 'warped_cloth')
        refined_cloth_dir = os.path.join(output_dir, 'refined_cloth')

        os.makedirs(fake_image_dir, exist_ok=True)
        os.makedirs(warped_cloth_dir, exist_ok=True)
        os.makedirs(refined_cloth_dir, exist_ok=True)

        for j in range(opt.batchSize):
            print("Saving", data['name'][j])

            # Get full paths
            fake_path = os.path.join(fake_image_dir, data['name'][j])
            warped_path = os.path.join(warped_cloth_dir, data['name'][j])
            refined_path = os.path.join(refined_cloth_dir, data['name'][j])

            # Ensure directories exist
            os.makedirs(os.path.dirname(fake_path), exist_ok=True)
            os.makedirs(os.path.dirname(warped_path), exist_ok=True)
            os.makedirs(os.path.dirname(refined_path), exist_ok=True)

            # Save images
            util.save_tensor_as_image(fake_image[j], fake_path)
            util.save_tensor_as_image(warped_cloth[j], warped_path)
            util.save_tensor_as_image(refined_cloth[j], refined_path)


if __name__ == '__main__':
    main()
