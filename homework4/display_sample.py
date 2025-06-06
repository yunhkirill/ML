from utils import read_images, extract_sample, display_sample

trainx, trainy = read_images('data/images_background')

sample_example = extract_sample(8, 5, 5, trainx, trainy)
display_sample(sample_example['images'])