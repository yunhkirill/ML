import os

import numpy as np


def read_alphabets(alphabet_directory_path):
    """
    Reads all the characters from a given alphabet_directory
    Args:
      alphabet_directory_path (str): path to diretory with files
    Returns:
      datax (np.array): array of path name of images
      datay (np.array): array of labels
    """
    datax = []  # all file names of images
    datay = []  # all class names

    character_dirs = os.listdir(alphabet_directory_path)

    for character_dir in character_dirs:
        character_path = os.path.join(alphabet_directory_path, character_dir)

        if not os.path.isdir(character_path):
            continue

        image_files = os.listdir(character_path)

        for image_file in image_files:
            if image_file.endswith('.png'):
                image_path = os.path.join(character_path, image_file)
                datax.append(image_path)
                datay.append(character_dir)

    return np.array(datax), np.array(datay)


def read_images(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None

    results = [read_alphabets(base_directory + '/' + directory + '/') for directory in os.listdir(base_directory)]

    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.concatenate([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay


