import os
import random 

def create_image_lists(dir_raw_images, fraction_train = 0.5, fraction_validation = 0.25):
    file_list = os.listdir(dir_raw_images)

    if (fraction_train + fraction_validation >= 1):
        print("fraction_train + fraction_validation is > 1!")
        print("setting fraction_train = 0.5, fraction_validation = 0.25")
        fraction_train = 0.5
        fraction_validation = 0.25
        
    fraction_test = 1 - fraction_train - fraction_validation

    image_list = [x for x in file_list if x.endswith("png") ]

    random.shuffle(image_list)

    index_train_end = int( len(image_list) * fraction_train)
    index_validation_end = index_train_end + int(len(image_list) * fraction_validation)

    # split into two parts for training and testing 
    image_list_train = image_list[0:index_train_end]
    image_list_test = image_list[index_train_end:(index_validation_end)]
    image_list_validation = image_list[index_validation_end:]
    return(image_list_train, image_list_test, image_list_validation)


def write_path_files(file_path, list):
    with open(file_path, 'w') as myfile:
        for line in  list: myfile.write(line + '\n')

