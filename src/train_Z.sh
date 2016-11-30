python facenet_train_classifier.py --gpu_memory_fraction .75 --logs_base_dir ~/logs/facenet/ --models_base_dir ~/facenet/ --data_dir ~/Documents/fer2013/all_yall/  --image_size 224 --model_def models.inception_resnet_v1 --batch_size 30  --weight_decay 2e-4 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --epoch_size 150 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file ../data/learning_rate_schedule_classifier_long.txt --center_loss_factor 2e-5 
 

