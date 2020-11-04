import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

from datetime import timedelta, datetime, date
import pandas as pd
import os
import shutil
import sqlalchemy

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
database_url = 'postgresql+psycopg2://login:password@localhost:5432/mydb'

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    length = 0
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
            vid = cv2.VideoCapture(FLAGS.video)
    print('Opened video ', FLAGS.video, '. W x H ', int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), ' x ',
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None

    dir_prefix = os.path.splitext(FLAGS.video)[0]
    vid_name = dir_prefix.replace('static/', '')
    if os.path.exists(dir_prefix):
        shutil.rmtree(dir_prefix)
        os.mkdir(dir_prefix)
    else:
            os.mkdir(dir_prefix)
    dir_prefix += '/'


    out = None
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1
        start_time = datetime.min
        bike_set = set()
        bike_list = []

        bike_dict = {'frame': 0, 'id': 0, 'finish_time': start_time, 'Recognitions': 'default', 'Recognised_plate': 'default', 'plate_number': 'XXX',  # 'appearance_num':0,
                     'Bike_image': None, 'Full_frame': None, 'Image_name': 'default', 'Full_image_name': 'default'}


        # This is needed to write image fragments on disk
        img_name = 'default'
        full_img_name = 'default'



    fps = 0.0
    count = 0
    frame_num = -1
    while True:
        _, img = vid.read()
        frame_num += 1
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name != 'bicycle':
                continue
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # Routines for bike detection at finish
            if not (track.track_id in bike_set):
                bike_set.add(track.track_id)
                input_fps = int(vid.get(cv2.CAP_PROP_FPS))
                dt = start_time + timedelta(seconds=frame_num / input_fps)
                bike_list.append(dict(bike_dict, frame=frame_num, finish_time=str(dt.time()), id=track.track_id))
            else:
                # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                for item in bike_list:
                    if item['id'] == track.track_id and 600 < bbox[3] < 800:
                        # Clear previously written images, something like overwrite them
                        previous_image_name = item['Image_name']
                        previous_full_image_name = item['Full_image_name']
                        if os.path.exists(previous_image_name):
                            os.remove(previous_image_name)
                        if os.path.exists(previous_full_image_name):
                            os.remove(previous_full_image_name)
                        # Edit finish_time
                        input_fps = int(vid.get(cv2.CAP_PROP_FPS))
                        dt = start_time + timedelta(seconds=frame_num / input_fps)
                        item['finish_time'] = str(dt.time())[:-3]
                        # Save image and put to bike dict
                        img_name = str(frame_num) + '_' + str(item['id']) + '.jpg'
                        full_img_name = str(frame_num) + '_fullframe.jpg'
                        cv2.imwrite(dir_prefix + img_name, img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                        cv2.imwrite(dir_prefix + full_img_name, img)
                        item['Bike_image'] = '<img src="../static/' + vid_name + '/' + img_name + '" width="200" >'
                        item['Full_frame'] = '<img src="../static/' + vid_name + '/' + full_img_name + '" width="500" >'
                        item['Recognitions'] = '<img src="../static/' + vid_name + '/' + 'res_' + img_name + '" width="200" >'
                        item['Recognised_plate'] = '<img src="../static/' + vid_name + '/' + 'plate_' + img_name + '" width="200" >'
                        item['Image_name'] = dir_prefix + img_name
                        item['Full_image_name'] = dir_prefix + full_img_name
                        # Update appearance and frame_num
                        # item['appearance_num']+=1
                        item['frame'] = frame_num
                        break

                    ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
                    # for det in detections:
                    #    bbox = det.to_tlbr()
                    #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

                    # print fps on screen and to commandline
                fps = (fps + (1. / (time.time() - t1))) / 2
                if frame_num % 15 == 0:
                    time_left = int((length - frame_num) / fps)
                

                # if FLAGS.output:
                # out.write(img)
                # frame_index = frame_index + 1
                # list_file.write(str(frame_index)+' ')
                # if len(converted_boxes) != 0:
                #    for i in range(0,len(converted_boxes)):
                #        list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1])
                #                        + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
                # list_file.write('\n')

        # press q to quit
        


    vid.release()
    if FLAGS.output:
        out.release()

        df = pd.DataFrame(bike_list)
        df = df.dropna()
        if df.empty:
            print('The resulting dataframe is empty')
        else:
            del df['Image_name']
            del df['Full_image_name']
            df = df.sort_values(by='finish_time', ascending=True)
            df.reset_index(drop=True)


            engine = sqlalchemy.create_engine(database_url)
            table_name = 'table_' + vid_name
            df.to_sql(table_name, engine)


        list_file.close()





if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
