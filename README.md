# Object detection API
### Setup tensoflow MODELS
#### Install neccessary python modules
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk 
conda install tensorflow-gpu==1.14
pip install Cython contextlib2 jupyter matplotlib cython pycocotools
```
#### Install tensorflow object detection API
```
wget https://github.com/tensorflow/models/archive/v1.13.0.zip
unzip v1.13.0.zip 
mv models-1.13.0 models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cd ../..
export PYTHONPATH=$PYTHONPATH:$PWD/models/research/:$PWD/models/research/slim
```
### Configure Training Data
#### Download pretrained model to finetune
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz 
rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
#### Convert xml annotations to csv: 
```
python xml_to_csv.py
```
#### Edit generate_tfrecord.py
```
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    else:
        None
```
#### Edit training/labelmap.pbtxt
```
item {
  id: 1
  name: 'basketball'
}

item {
  id: 2
  name: 'shirt'
}
```

#### Generate TFRecords:
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```


#### Edit training/faster_rcnn_inception_v2_pets.config

- Line 9. Change num_classes : 3 .
  ```
  num_classes:3
  ```
  
- Line 130. Change num_examples to the number of images in the /images/test directory.
   ```
   num_examples:100
   ```
#### Change below paths if any error comes
- Line 106. Change fine_tune_checkpoint to:
   ```
   fine_tune_checkpoint : "./faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
   ```

- Lines 123 and 125. change input_path and label_map_path to:
   ```
   input_path : "./train.record"
   label_map_path: "./training/labelmap.pbtxt" 
   ```
   
- Lines 135 and 137. Change input_path and label_map_path to:
  ```
  input_path : "./test.record"
  label_map_path: "./training/labelmap.pbtxt"
  ```

### Run the Training

#### Command to start training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

#### Command to open tensorboard:
```
tensorboard --logdir=training
```

### Export Inference Graph
#### Remove inference_graph folder if exists
#### Replace XXXX in below command
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
