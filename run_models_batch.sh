python3 check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt > resnet_pet-images.txt
python3 check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt
python3 check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt > vgg_pet-images.txt
