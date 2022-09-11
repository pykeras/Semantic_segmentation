import xml.etree.ElementTree as ET, numpy as np
import cv2, os, glob
from skimage import draw
from tqdm import tqdm

train_annot_path = "./data/MoNuSeg 2018 Training Data/Annotations/"
train_dest = "./data/MoNuSeg 2018 Training Data/Masks"

test_image_path = "./data/MoNuSegTestData/"
test_dest = "./data/MoNuSegTestData/Masks"

train_annot = glob.glob(train_annot_path+"*.xml")
test_annot = glob.glob(test_image_path+"*.xml")

def binary_mask(annot_files, dest_path):
  
  os.makedirs(dest_path, exist_ok=True)
  
  for xml_file in tqdm(annot_files, total=len(annot_files)):
    name = xml_file.split("/")[-1].split(".")[0]
    child = ET.parse(xml_file).getroot()[0]
    for i in child:
      i_t = i.tag
      binary_mask = np.transpose(np.zeros((1000, 1000)))
      if i_t == 'Regions':
        for j in i:
          j_t = j.tag
          if j_t == 'Region':
            regions, vertices = [], j[1]
            coords = np.zeros((len(vertices), 2))
            for idx, vertex in enumerate(vertices):
              coords[idx][0] = vertex.attrib['X']
              coords[idx][1] = vertex.attrib['Y']
            regions.append(coords)
            vertex_row_coords = regions[0][:,0]
            vertex_col_coords = regions[0][:,1]
            fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, binary_mask.shape)
            binary_mask[fill_row_coords, fill_col_coords] = 255
          
        mask_path = f'{dest_path}/{name}.png'
        cv2.imwrite(mask_path, binary_mask)
        
        
        
binary_mask(train_annot, train_dest)
binary_mask(test_annot, test_dest)