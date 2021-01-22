import cv2

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from config_lib import make_config_detector, make_config_symbols

import re
import math

prefix = '/content/drive/My Drive/Studying/Netology/Diploma'

region_arr = [45, 12, 123, 76, 163, 116, 31, 66, 164, 89, 
                      159, 85, 36, 11, 64, 74, 190, 23, 43, 71, 17, 32, 
                      154, 127, 142, 8, 77, 33, 88, 29, 5, 16, 22, 14, 125, 
                      41, 55, 54, 102, 173, 138, 46, 91, 18, 82, 75, 96, 94, 
                      13, 81, 62, 80, 67, 51, 56, 40, 79, 38, 44, 1, 134, 39, 28, 
                      113, 95, 178, 97, 27, 10, 73, 58, 6, 136, 57, 72, 35, 174, 9, 
                      70, 34, 86, 26, 7, 24, 750, 121, 68, 197, 53, 196, 177, 52, 61, 
                      19, 186, 2, 4, 93, 199, 25, 37, 87, 152, 30, 47, 84, 48, 126, 65, 
                      83, 60, 21, 124, 99, 92, 98, 63, 49, 15, 161, 150, 69, 90, 42, 777, 
                      3, 59, 50, 169, 78]

class ImagePrediction(object):
    def __init__(self, img_path, predictor):
      self.path = img_path
      self.predictor = predictor
      self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
      self.prediction = predictor(self.img)
      self.pred_boxes = self.prediction['instances'].get('pred_boxes').to("cpu")
      self.pred_scores = self.prediction['instances'].get('scores').to("cpu")

      self.cropped_plates = self.crop_carplates()

      self.tesseract_res = self.get_tesseract_results()
      self.custom_res = self.get_custom_result()
      self.semi_custom_res = self.get_semi_custom_result()

    def crop_carplates(self):
      output = []
      for r in self.pred_boxes:
        y_min = math.floor(r[0])
        y_max = math.ceil(r[2])
        x_min = math.floor(r[1])
        x_max = math.ceil(r[3])
        plate = self.img[x_min:x_max, y_min:y_max, :]
        output.append(plate)
      return output

    def visualize_prediction(self):
      image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
      for i, r in enumerate(self.pred_boxes):
          max_l = max([len('Tesseract:  '+self.tesseract_res[i]), 
                       len('CustomNN:  '+self.custom_res[i])])
          image = cv2.rectangle(image, (r[0],r[1]), (r[2], r[3]), (255, 0, 0), 2) 
          image = cv2.rectangle(image, (r[0],r[1]-70), (r[0]+15*max_l+10, r[1]), (240, 255, 240), -1)
          cv2.putText(image,str(round(self.pred_scores[i].item()*100))+'%', 
                      (r[0],r[3]+30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8,
                      (255, 0, 0),
                      2,
                      cv2.LINE_AA)
          cv2.putText(image,'Tesseract:  '+self.tesseract_res[i], 
                      (r[0],r[1]-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8,
                      (0, 0, 0),
                      2,
                      cv2.LINE_AA)
          cv2.putText(image,'CustomNN:  '+self.custom_res[i], 
                      (r[0],r[1]-35), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8,
                      (0, 0, 0),
                      2,
                      cv2.LINE_AA)
      return image

    def get_tesseract_results(self):
      result = []
      for p in self.cropped_plates:
        tes_img = self.prepare_img_for_tesseract(p)
        res = pytesseract.image_to_string(tes_img, 
                                                  config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCEHKMOPTXY0123456789')
        res = res.replace(' ', '')
        res = re.findall('\w*', res)[0]
        result.append(res)
      return result

    def get_custom_result(self):
      cfg_sym = get_cfg()
      cfg_sym = make_config_symbols(cfg_sym, prefix)

      result = []
      predictor_symbols = DefaultPredictor(cfg_sym)
      for p in self.cropped_plates:
        out = predictor_symbols(p)
        res = self.make_string_from_pred_1(out)
        if len(res)>7:
          part_1 = res[0].replace('8','B').replace('9','P').replace('0','O')
          part_2 = res[1:4].replace('B','8').replace('P','9').replace('O','0')
          part_3 = res[4:6].replace('8','B').replace('9','P').replace('0','O')
          part_4 = res[6:].replace('B','8').replace('P','9').replace('O','0')
          if part_4.isdigit():
            if int(part_4) in region_arr:
              result.append(part_1+part_2+part_3+part_4)
            elif int(part_4[:-1]) in region_arr:
              part_4 = part_4[:-1]
              result.append(part_1+part_2+part_3+part_4)
            else:
              result.append(part_1+part_2+part_3+part_4)
          else:
            result.append(part_1+part_2+part_3+part_4)
        else:
          result.append(res)
      return result
      
    def get_semi_custom_result(self):
      result = []
      cfg_sym = get_cfg()
      cfg_sym = make_config_symbols(cfg_sym, prefix)
      predictor_symbols = DefaultPredictor(cfg_sym)
      for p in self.cropped_plates:
        out = predictor_symbols(p)
        rects = out["instances"].get('pred_boxes').to("cpu")
        pred_classes = out["instances"].get('pred_classes').to("cpu")
        pred_scores = out["instances"].get('scores').to("cpu")
        cropped_symbols = self.crop_symbols(p, rects, pred_classes,pred_scores)
        # ###################################################
        number_string = ''
        for i, s in enumerate(cropped_symbols):
          if i in [1,2,3,6,7,8]:
            symbol = self.recognize_symbol(s, '0123456789')[0]
          elif i in [0,4,5]:
            symbol = self.recognize_symbol(s, 'ABCEHKMOPTXY')[0]
          number_string += symbol
        result.append(number_string)
      return result


          
    def make_string_from_pred_1(self, outputs):
      dict_ = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', 
                    '10':'A', '11':'B', '12':'C', '13':'E', '14':'H', '15':'K', '16':'M', '17':'O',
                    '18':'P', '19':'T', '20':'X', '21':'Y'}
      pred_boxes = outputs["instances"].get('pred_boxes').get_centers().to("cpu")
      pred_scores = outputs["instances"].get('scores').to("cpu")
      pred_classes = outputs["instances"].get('pred_classes').to("cpu")
      d = {}
      s = {}
      for i in range(len(pred_classes)):
        d[i] = float(pred_boxes[i][0])
        s[i] = float(pred_scores[i])
      sorted_list = [(i,d[i],s[i]) for i in sorted(d, key=d.get)]

      l=[sorted_list[0]]
      for i in range(len(sorted_list)):
        if ((sorted_list[i][1]-l[-1][1]) < 5):
          if ((sorted_list[i][2]-l[-1][2]) <= 0):
            continue
          else:
            l[-1] = sorted_list[i]
        else:
          l.append(sorted_list[i])

      str_ = ''
      for i in [i[0] for i in l]:
        str_ += dict_[str(int(pred_classes[i]))] 

      return str_

    def crop_symbols(self, img, boxes, pred_classes,pred_scores):
      d = {}
      s = {}
      for i in range(len(pred_classes)):
        d[i] = float(boxes.get_centers()[i][0])
        s[i] = float(pred_scores[i])
      sorted_list = [(i,d[i],s[i]) for i in sorted(d, key=d.get)]

      l=[sorted_list[0]]
      for i in range(len(sorted_list)):
        if ((sorted_list[i][1]-l[-1][1]) < 5):
          if ((sorted_list[i][2]-l[-1][2]) <= 0):
            continue
          else:
            l[-1] = sorted_list[i]
        else:
          l.append(sorted_list[i])

      symbols_imgs = []
      for r in boxes:
        y_min = math.floor(r[0])
        y_max = math.ceil(r[2])
        x_min = math.floor(r[1])
        x_max = math.ceil(r[3])
        symbols_imgs.append(img[x_min:x_max, y_min:y_max, :])
        symbols_imgs_sorted = []
      last = 0
      for i in [i[0] for i in l]:
        symbols_imgs_sorted.append(symbols_imgs[i])
      
      return symbols_imgs_sorted

    
    def recognize_symbol(self, sym, sym_dict):
      sym_gray = self.prepare_img_for_tesseract(sym)
      sym_str = pytesseract.image_to_string(sym_gray, 
                                            config = '--psm 8 --oem 3 -c tessedit_char_whitelist={}'.format(sym_dict))
      return sym_str

    def enlarge_img(self, image, scale_percent):
      width = int(image.shape[1] * scale_percent / 100)
      height = int(image.shape[0] * scale_percent / 100)
      dim = (width, height)
      resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)    
      return resized_image 

    def prepare_img_for_tesseract(self, img):
      img = self.enlarge_img(img, 150)
      gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      return gray