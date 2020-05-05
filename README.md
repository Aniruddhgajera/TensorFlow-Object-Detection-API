# Object detection API
### MODELS
```
conda install tensorflow-gpu==1.14
wget https://github.com/tensorflow/models/archive/v1.13.0.zip
unzip v1.13.0.zip 
mv models-1.13.0 models

sudo apt-get install protobuf-compiler python-pil python-lxml python-tk cython pycocotools

pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib

cd models/research
protoc object_detection/protos/*.proto --python_out=.
cd ../..
export PYTHONPATH=$PYTHONPATH:$PWD/models/research/:$PWD/models/research/slim
```
### 4. Generate Training Dataith some slight modifications to work with our directory structure.

#### Step1: 
```
python xml_to_csv.py
```
#### Step2:
##### (1) Edit generate_tfrecord.py
```
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        None
```
##### (2) Edit training/labelmap.pbtxt
```
item {
  id: 1
  name: 'nine'
}

item {
  id: 2
  name: 'ten'
}
```

##### (3) Generate TFRecords:
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```


#### 5b. Configure training

- Line 9. Change num_classes : 3 .

- Line 106. Change fine_tune_checkpoint to:
  - fine_tune_checkpoint : "./faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

- Lines 123 and 125. change input_path and label_map_path to:
  - input_path : "./train.record"
  - label_map_path: "./training/labelmap.pbtxt"

- Line 130. Change num_examples to the number of images in the /images/test directory.

- Lines 135 and 137. Change input_path and label_map_path to:
  - input_path : "./test.record"
  - label_map_path: "./training/labelmap.pbtxt"



### 6. Run the Training
**UPDATE 9/26/18:** 

Here we go! From the \object_detection directory, issue the following command to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

tensorboard --logdir=training
```

### 7. Export Inference Graph
Remove inference_graph folder if exists
Replace XXXX in below command
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
