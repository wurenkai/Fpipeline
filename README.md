# CKD-CFPUWF

## Preprocessing

Please follow the process in Figure 4 of the paper and use the code in the preprocessing folder and the calibration board to preprocess the images.

## Training

#### 1) Transformer

```
cd Transformer
python main.py --name [exp_name] --ckpt_path [save_path] \
               --data_path [training_image_path] \
               --validation_path [validation_image_path] \
               --mask_path [mask_path] \
               --BERT --batch_size 64 --train_epoch 100 \
               --nodes 1 --gpus 8 --node_rank 0 \
               --n_layer [transformer_layer #] --n_embd [embedding_dimension] \
               --n_head [head #] --ImageNet --GELU_2 \
               --image_size [input_resolution]
```

Notes of transformer: 
+ `--AMP`: Reduce the memory cost while training, but sometimes will lead to NAN.
+ `--use_ImageFolder`: Enable this option while training on ImageNet
+ `--random_stroke`: Generate the mask on-the-fly.
+ Our code is also ready for training on multiple machines.

#### 2) Guided Upsampling

```
cd Guided_Upsample
python train.py --model 2 --checkpoints [save_path] \
                --config_file ./config_list/config_template.yml \
                --Generator 4 --use_degradation_2
```

Notes of guided upsampling: 
+ `--use_degradation_2`: Bilinear downsampling. Try to match the transformer training.
+ `--prior_random_degree`: Stochastically deviate the sequence elements by K nearest neighbour.
+ Modify the provided config template according to your own training environments.
+ Training the upsample part won't cost many GPUs.


## Inference

We provide very covenient and neat script for inference.
```
python run.py --input_image [test_image_folder] \
              --input_mask [test_mask_folder] \
              --sample_num 1  --save_place [save_path] \
              --ImageNet --visualize_all
```

Notes of inference: 
+ `--sample_num`: How many completion results do you want?
+ `--visualize_all`: You could save each output result via disabling this option.
+ `--ImageNet` `--FFHQ` `--Places2_Nature`: You must enable one option to select corresponding ckpts.
+ Please use absolute path.

## CKD screening model

1. In the CKD screening model folder, specify the CSV file address and image address.
2. The CSV file must contain two columns, image_name and label. A label of 1 indicates a diagnosis of CKD, and 0 indicates normal.
3. After adding the addresses of each file as required, run the following command：
```
python train.py
```
4. For inference and testing, you need to make the same changes to the configuration file inside, and run the following command:
```
python test.py
```

## Acknowledgement
Thanks to [ICT](https://github.com/raywzy/ICT), [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [UNet++](https://github.com/MrGiovanni/UNetPlusPlus) for their outstanding work.
