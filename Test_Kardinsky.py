import torch.cuda
from kandinsky3 import get_T2I_Flash_pipeline
from kandinsky2 import get_kandinsky2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
import math
HEIGHT = 1024
WIDTH = 1024
HYBRID = False
MODES = ["reconstruct", "seperation", "building"]
MODE = MODES[1]
PREFIX = "dog_more"
import torch
CUDA = torch.cuda.is_available()
def convert_to_specific_size(image, h, w):
    return image.resize((w, h))
def get_squared_image(image):
    shape = image.size
    if shape[0] == shape[1]:
        return image
    if shape[0] > shape[1]:
        image = convert_to_specific_size(image, shape[1], shape[1])
    else:
        image = convert_to_specific_size(image, shape[0], shape[0])
    return image
def gen_image_Kand3(prompt, seed, output_file,output_dir="results/"):
    device_map = torch.device('cuda:0')
    dtype_map = {
        'unet': torch.float32,
        'text_encoder': torch.float16,
        'movq': torch.float32,
    }

    t2i_pipe = get_T2I_Flash_pipeline(
        device_map, dtype_map
    )
    res = t2i_pipe(prompt, seed=seed)[0]
    res.save(os.path.join(output_dir, output_file))
    return res
def gen_image_Kand2(model, prompt, seed, output_file,output_dir="results/"):
    images = model.generate_text2img(
        prompt,
        seed,
        num_steps=150,
        batch_size=1,
        guidance_scale=4,
        h=HEIGHT, w=WIDTH,
        sampler='p_sampler',
        prior_cf_scale=4,
        prior_steps="5"
    )
    images[0].save(os.path.join(output_dir, output_file))
    return images[0]
def create_collage(rows_of_images, output_file, text=None):
    # Determine the size of the new image
    max_widths = max(sum(img.width for img in row) for row in rows_of_images)
    max_images = max(len(row) for row in rows_of_images)
    max_heights = [max(img.height for img in row) for row in rows_of_images]
    num_rows = len(rows_of_images)
    new_width = max_widths + 20 * (max_images - 1)  # Extra space for the spacing between images

    # Calculate the maximum height across all rows
    max_height_row = max(max_heights)

    # Calculate the total height considering the maximum height and spacing between rows
    new_height = max_height_row * num_rows + 25 * (num_rows - 1) + 50 # Extra space for the text

    # Create a new blank image
    new_image = Image.new('RGB', (new_width, new_height), color='white')

    # Paste images in each row
    y_offset = 0
    for i, row in enumerate(rows_of_images):
        x_offset = 0
        if len(row) < max_images:
            x_offset = (max_widths - sum(img.width for img in row))//2+ 20 * math.ceil((max_images - len(row) - 1)/2)
        for img in row:
            #paste the image centered in the row
            new_image.paste(img, (x_offset , y_offset))
            x_offset += img.width
            x_offset += 20  # Add spacing between images
        y_offset += max_height_row + 25

    # Add text to the image
    if text is not None:
        font = ImageFont.load_default(size=72)  # You can change the font and size as needed
        draw = ImageDraw.Draw(new_image)
        text_position = (new_width // 2 - 5, new_height - 5)
        #draw the text in big font
        draw.text(text_position, text, fill='black', font=font, align='center', anchor='ms')

    # Save the new image
    new_image.save(output_file)
def create_image(first_row_images, second_row_images, output_file,text=None):
    # Determine the size of the new image
    max_width_first_row = max(img.width for img in first_row_images)
    max_width_second_row = max(img.width for img in second_row_images)
    max_height = max(img.height for img in first_row_images + second_row_images)
    new_width = max(max_width_first_row, max_width_second_row) * max(len(first_row_images), len(second_row_images))
    new_height = max_height * 2 + 50  # Extra space for the text

    # Create a new blank image
    new_image = Image.new('RGB', (new_width, new_height), color='white')

    # Paste images in the first row
    x_offset_first_row = (new_width - max_width_first_row * len(first_row_images)) // 2
    for img in first_row_images:
        new_image.paste(img, (x_offset_first_row, 0))
        x_offset_first_row += max_width_first_row

    # Paste images in the second row
    x_offset_second_row = (new_width - max_width_second_row * len(second_row_images)) // 2
    for img in second_row_images:
        new_image.paste(img, (x_offset_second_row, max_height))
        x_offset_second_row += max_width_second_row

    # Add text to the image
    if text is not None:
        font = ImageFont.load_default()  # You can change the font and size as needed
        draw = ImageDraw.Draw(new_image)
        text_position = (new_width // 2 - 50, new_height - 50)
        draw.text(text_position, text, fill='black', font=font)
# Save the new image
    new_image.save(output_file)
def create_gif(image_list, output_file, duration=0.1, loop=0,cyclic=True):
    """
    Create a GIF from a list of PIL images.

    Parameters:
        image_list (list): List of PIL images.
        output_file (str): Output filename for the GIF.
        duration (float): Duration (in seconds) for each frame in the GIF. Default is 0.2 seconds.
    """
    # List of durations for each frame (in seconds)
    if cyclic:
        # Duplicate the image list and concatenate it with itself in reverse order
        image_list = image_list + list(reversed(image_list[:-1]))

        # List of durations for each frame (in seconds)
        durations = [duration for _ in image_list]  # Adjust as needed
    else:
        durations = [duration for i in range(len(image_list))]  # Adjust as needed
    durations[0] = 2.5
    durations[-1] = 2.5

    # Save the GIF with specified durations
    imageio.mimsave(output_file, image_list, duration=durations, loop=loop)
def reconstruct(images_dir, seeds=[3,15,20,35,40,43,47,50]):
    if CUDA:
        model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
    else:
        model = get_kandinsky2('mps', task_type='text2img', model_version='2.1', use_flash_attention=False)
    #Get all the images in the directory
    images_texts = [get_squared_image(Image.open(os.path.join(images_dir, f))) for f in os.listdir(images_dir) if not f.startswith('.')]
    variation = {"num_steps": 150, "h": HEIGHT, "w": WIDTH, "PCA_components": 0, "which_to_keep": []}
    #Iterate over all the images in the directory
    i = 0
    #Clean Cache of GPU
    torch.cuda.empty_cache()
    for image in images_texts:
        temp_list = []
        for seed in seeds:
            torch.cuda.empty_cache()
            # Construct the title based on parameter values
            title = f'{i}_Reconstruct-steps={variation["num_steps"]}, h={variation["h"]}, w={variation["w"]}'
            #Add seed to the title seed={seed}
            title += f',seed={seed}'
            #add weights to the title
            # fig.suptitle(title, fontsize=8)
            images = model.mix_images2(
                [image],
                [1.0],
                num_steps=variation["num_steps"],
                batch_size=1,
                guidance_scale=5,
                h=variation["h"],
                w=variation["w"],
                sampler='p_sampler',
                prior_cf_scale=4,
                prior_steps="5",
                negative_prior_prompt="",
                negative_decoder_prompt=variation.get("negative_decoder_prompt", ""),
                PCA_components=variation.get("PCA_components", 0),
                which_to_keep=variation.get("which_to_keep", []),
                seed=seed
            )
            temp_list.append(images[0])
            # Create the image with all the results
        create_image([convert_to_specific_size(image, HEIGHT, WIDTH)], temp_list, f"Rec_Res/{i}_h{variation['h']}_w{variation['w']}_seed{seed}_steps{variation['num_steps']}.png")
        i += 1
    return
def gen_variations(seed=-1, which_to_keep=-1, which_to_reconstruct=-1):
    if which_to_reconstruct != -1:
        plot_variations = [
            {"num_steps": 150, "h": HEIGHT, "w": WIDTH, "PCA_components": len(images_texts), "which_to_keep": [], \
             "which_to_reconstruct": [i], "seed": i + 1} for i in range(len(images_texts) - 1)]
    return 0
def unique_variations(plot_variations, len_images_texts):
    # plot_variations = [{"num_steps": 150, "h": HEIGHT, "w": WIDTH, "PCA_components": i, "which_to_keep": []} for i in range(1, 17)]
    # plot_variations = [{"num_steps": 150, "h": HEIGHT, "w": WIDTH, "PCA_components": 0, "which_to_keep": []}]
    plot_variations_unique = []
    plot_variations_set = set()

    for variation in plot_variations:
        # Convert 'which_to_keep' to tuple for hashing
        variation_tuple = (variation["num_steps"], variation["h"], variation["w"], variation["PCA_components"],
                           tuple(variation["which_to_keep"]))

        # Check if the variation is unique
        if variation_tuple not in plot_variations_set:
            plot_variations_set.add(variation_tuple)
            plot_variations_unique.append(variation)
    print(plot_variations_unique)
    return plot_variations_unique

def exp_Amp_PCA_AVG(seed=43):
    data_dir = "cat_prompts/"
   # data_dir = "test_cats2/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    result_dir = "EXP_amp_avg/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if CUDA:
        model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
    else:
        model = get_kandinsky2('mps', task_type='text2img', model_version='2.1', use_flash_attention=False)
    # images_texts = [gen_image_Kand2(model, dog_prompts[i], seed, f"dog_{i}.png", data_dir) for i in range(len(dog_prompts))]
    images_texts = [convert_to_specific_size(Image.open(os.path.join(data_dir, f)), HEIGHT, WIDTH) for f in os.listdir(data_dir) if not f.startswith('.')]
    weights_list = [[1 / len(images_texts) for i in range(len(images_texts))]]
    plot_variations = [
        {"num_steps": 150, "h": HEIGHT, "w": WIDTH, "PCA_components": len(images_texts), "which_to_keep": [], "seed": seed}]
    which_to_amp = [[i] for i in range(len(images_texts))]
    amps = [1, 1.5, 2, 5, 7.5, 10, 50, -1, -2, -10]
    amps = [1, 1.1, 1.3, 1.5, 2, 5, -1, -1.5, -2]
 #   amps = [1,  10]
    final_list = [images_texts]
    for amp in amps:
        temp_list = []
        for keep_idx in which_to_amp:
            print(f"Start Working on Avg with amplified image with keep_idx={keep_idx}, seed={seed}, amp={amp}")
            # Save Average Image
            for idx, weights in enumerate(weights_list):
                avg_image = model.gen_avg_amp(
                    images_texts,
                    weights,
                    num_steps=plot_variations[0]["num_steps"],
                    batch_size=1,
                    guidance_scale=5,
                    h=plot_variations[0]["h"],
                    w=plot_variations[0]["w"],
                    sampler='p_sampler',
                    prior_cf_scale=4,
                    prior_steps="5",
                    negative_prior_prompt="",
                    negative_decoder_prompt=plot_variations[0].get("negative_decoder_prompt", ""),
                    PCA_components=plot_variations[0].get("PCA_components", 0),
                    which_to_keep=plot_variations[0].get("which_to_keep", []),
                    seed=plot_variations[0].get("seed", 43),
                    which_to_amp=keep_idx,
                    amplify=amp
                )
                temp_list.append(avg_image[0])
        print("The length of the temp_list is ", len(temp_list))
        final_list.append(temp_list)
    create_collage(final_list, f"{result_dir}{PREFIX}_collage_{data_dir.replace('/', '')}.png")


    return

def exp_check_rec2(seed=43):
    data_dir = "toy_soldier/"
    data_dir = "dogs_prompts/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    result_dir = "Amp/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if CUDA:
        model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
    else:
        model = get_kandinsky2('mps', task_type='text2img', model_version='2.1', use_flash_attention=False)
   # images_texts = [gen_image_Kand2(model, dog_prompts[i], seed, f"dog_{i}.png", data_dir) for i in range(len(dog_prompts))]
    images_texts = [convert_to_specific_size(Image.open(os.path.join(data_dir, f)), HEIGHT, WIDTH) for f in os.listdir(data_dir) if not f.startswith('.')]
    weights_list = [[1 / len(images_texts) for i in range(len(images_texts))]]
    if MODE == "seperation":
        which_to_keep = [[]] + [[i] for i in range(0,8)]
    if MODE == "building":
        which_to_keep = [[]]
        for l in range(0, 15):
            which_to_keep.append([k for k in range(0, l + 1)])
    plot_variations = [{"num_steps": 150, "h": HEIGHT, "w":   WIDTH, "PCA_components": len(images_texts), "which_to_keep": [], \
                        "which_to_reconstruct": [i], "seed": seed} for i in range(len(images_texts))]
   # plot_variations = unique_variations(plot_variations, len(images_texts))
    final_list = [images_texts]
    avgs = []
    amps = [1, 1.5, 2, 5, 7.5, 10, 50, -1, -2, -10]
    amps = [1.5, 2, 5, 7.5, 10, 50, -2, -1, 1]
    amps = [1, 1.2, 1.5, 2, 3, -1]
    which_to_keep = [[]]
    which_to_amplifys = [[i] for i in range(0, 3)]
    for amp in amps:
        print(f"Start working on Collage with amp={amp}, seed={seed}")
        temp_list = []
        for which_to_amplify in which_to_amplifys:
            print("Which to amplify is ", which_to_amplify)
            for variation in plot_variations:
                variation["which_to_keep"] = []
                for idx, weights in enumerate(weights_list):
                    images = model.mix_images_reconstruct(
                        images_texts,
                        weights,
                        num_steps=variation["num_steps"],
                        batch_size=1,
                        guidance_scale=5,
                        h=variation["h"],
                        w=variation["w"],
                        sampler='p_sampler',
                        prior_cf_scale=4,
                        prior_steps="5",
                        negative_prior_prompt="",
                        negative_decoder_prompt=variation.get("negative_decoder_prompt", ""),
                        PCA_components=variation.get("PCA_components", 0),
                        which_to_keep=variation.get("which_to_keep", []),
                        which_to_reconstruct=variation.get("which_to_reconstruct", []),
                        seed=variation.get("seed", 43),
                        amplify = amp,
                        which_to_amplify = which_to_amplify
                    )
                    images[0].save(f"{result_dir}{PREFIX}_Amp_{amp}_which_to_amplify_{which_to_amplify}_which_to_reconstruct_{variation['which_to_reconstruct']}.png")
                    temp_list.append(images[0])
            final_list.append(temp_list)
        #Save Average Image
    create_collage(final_list, f"{result_dir}{PREFIX}_collage_keep_val_seperation_upto7{data_dir.replace('/', '')}.png")
        # Create the image with all the results
       # create_image([Image.open(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if not f.startswith('.')],
                  #   temp_list, f"{result_dir}collage_keep_val_{keep_idx}_{data_dir.replace('/', '')}.png")

    return
def exp_check_rec(seed=43):
    data_dir = "toy_soldier/"
    #data_dir = "dogs_prompts/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    result_dir = "Rec_Res_Results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if CUDA:
        model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
    else:
        model = get_kandinsky2('mps', task_type='text2img', model_version='2.1', use_flash_attention=False)
   # images_texts = [gen_image_Kand2(model, dog_prompts[i], seed, f"dog_{i}.png", data_dir) for i in range(len(dog_prompts))]
    images_texts = [Image.open(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if not f.startswith('.')]
    weights_list = [[1 / len(images_texts) for i in range(len(images_texts))]]
    if MODE == "seperation":
        which_to_keep = [[]] + [[i] for i in range(0,8)]
    if MODE == "building":
        which_to_keep = [[]]
        for l in range(0, 15):
            which_to_keep.append([k for k in range(0, l + 1)])
    plot_variations = [{"num_steps": 150, "h": HEIGHT, "w":   WIDTH, "PCA_components": len(images_texts), "which_to_keep": [], \
                        "which_to_reconstruct": [i], "seed": seed} for i in range(len(images_texts))]
   # plot_variations = unique_variations(plot_variations, len(images_texts))
    final_list = [images_texts]
    avgs = []
    for keep_idx in which_to_keep:
        print(f"Start working on Collage with keep_idx={keep_idx}, seed={seed}")
        temp_list = []
        for variation in plot_variations:
            variation["which_to_keep"] = keep_idx
            for idx, weights in enumerate(weights_list):
                images = model.mix_images_reconstruct(
                    images_texts,
                    weights,
                    num_steps=variation["num_steps"],
                    batch_size=1,
                    guidance_scale=5,
                    h=variation["h"],
                    w=variation["w"],
                    sampler='p_sampler',
                    prior_cf_scale=4,
                    prior_steps="5",
                    negative_prior_prompt="",
                    negative_decoder_prompt=variation.get("negative_decoder_prompt", ""),
                    PCA_components=variation.get("PCA_components", 0),
                    which_to_keep=variation.get("which_to_keep", []),
                    which_to_reconstruct=variation.get("which_to_reconstruct", []),
                    seed=variation.get("seed", 43)
                )
                temp_list.append(images[0])
        final_list.append(temp_list)
        #Save Average Image
        for idx, weights in enumerate(weights_list):
            avg_image = model.mix_images2(
                images_texts,
                weights,
                num_steps=plot_variations[0]["num_steps"],
                batch_size=1,
                guidance_scale=5,
                h=plot_variations[0]["h"],
                w=plot_variations[0]["w"],
                sampler='p_sampler',
                prior_cf_scale=4,
                prior_steps="5",
                negative_prior_prompt="",
                negative_decoder_prompt=plot_variations[0].get("negative_decoder_prompt", ""),
                PCA_components=plot_variations[0].get("PCA_components", 0),
                which_to_keep=plot_variations[0].get("which_to_keep", []),
                which_to_reconstruct=[],
                seed=plot_variations[0].get("seed", 43)
            )
            #avg_image[0].save(
            #    f"{result_dir}{PREFIX}_Average_image_for_{keep_idx}_{data_dir.replace('/', '')}.png")
            avgs.append(avg_image[0])
    final_list.append(avgs)
    create_collage(final_list, f"{result_dir}{PREFIX}_collage_keep_val_seperation_upto7{data_dir.replace('/', '')}.png")
        # Create the image with all the results
       # create_image([Image.open(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if not f.startswith('.')],
                  #   temp_list, f"{result_dir}collage_keep_val_{keep_idx}_{data_dir.replace('/', '')}.png")

    return



if __name__ == '__main__':
    # exp_Amp_PCA_AVG()
    # exit(0)
    # ##GENERATE BUILDING/SEPERATION IMAGES
    exp_check_rec2()
    exit(0)
    data_dir = "test_rec_"
    #Create the dir if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
