import pandas as pd

def generate_config():
    all_classes = pd.read_csv('images/train_labels.csv')['class'].unique()
    test_example_count = len(pd.read_csv('images/test_labels.csv'))-1
    print('Done: CSV parsed Unique classes: ',all_classes)

    num_steps = 15000

    generate_tf_record = ''

    f = open("training/labelmap.pbtxt", "w")
    for idx, value in enumerate(all_classes):

        first_class = '''    
        if row_label == 'first':
            return 1
                '''
        all_other_class = '''
        elif row_label == 'other':
            return 2
        '''    
        label_map = '''
item {
  id: 1
  name: 'first'
}
    '''

        if idx == 0:
            to_append = first_class.replace('first',value)
        else:
            to_append = all_other_class.replace('other',value).replace('2',str(idx+1))

        generate_tf_record+=to_append

        f.write(label_map.replace('first',value).replace('1',str(idx+1)))

    f.close()

    #print('\nGenerated string:\n',generate_tf_record)

    with open("generate_tfrecord_base.py") as f:
        lines = f.readlines()

    with open("generate_tfrecord.py", "w") as f:
        lines.insert(32, generate_tf_record)
        f.write("".join(lines))

    print('Done: Edited generate_tfrecord.py')
    print('Done: Edited labelmap.pbtxt')

    with open("training/faster_rcnn_inception_v2_pets_base.config") as f:
        lines = f.readlines()

    with open("training/faster_rcnn_inception_v2_pets.config", "w") as f:
        lines[8] = '    num_classes: {}\n'.format(str(len(all_classes)))
        lines[131] = '  num_examples: {}\n'.format(str(test_example_count))
        lines[115]= '  num_steps: {}\n'.format(str(num_steps))
        f.write("".join(lines))
        
    print('Done: Edited training/faster_rcnn_inception_v2_pets.config')