import torch
from PIL import Image, ImageFont, ImageDraw, ImageOps
import numpy as np
import time
import re
import folder_paths
import json
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import os


# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class TextTokens:
    def __init__(self):
        self.tokens = {
            '[time]': str(time.time()).replace('.', '_')
        }
        if '.' in self.tokens['[time]']: self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()

        # Update time
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            text = text.replace(token, value)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)
        return text


#------ Image Stuff
class BastardCropImageByMask:
    """
    A model loader.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE", ),
            "mask": ("IMAGE", ),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_checkpoint"

    #OUTPUT_NODE = False

    CATEGORY = "loaders"

    def load_checkpoint(self, image, mask):
        image = tensor2pil(image).convert('RGBA')
        mask = tensor2pil(mask).convert('L')
        
       # Find bounding box of the white portion in the mask
        bbox = mask.getbbox()
        
        # Crop the white portion from the original image
        cropped_img = image.crop(bbox)
        
        # Create an alpha layer from the mask, cropped to the bounding box
        alpha_layer = mask.crop(bbox)
        
        # Combine original cropped image and alpha layer
        cropped_img.putalpha(alpha_layer)
        cropped_img = pil2tensor(cropped_img)
        return (cropped_img,)

class BastardIsolateSubjectByMask:
    """
    A model loader.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE", ),
            "mask": ("IMAGE", ),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_checkpoint"

    #OUTPUT_NODE = False

    CATEGORY = "loaders"

    def load_checkpoint(self, image, mask):
        image = tensor2pil(image).convert('RGBA')
        mask = tensor2pil(mask).convert('L')
        
        # Combine original cropped image and alpha layer
        image.putalpha(mask)
        image = pil2tensor(image)
        return (image,)

class BastardImageOverImage:
    """
    A model loader.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "base_image": ("IMAGE", ),
            "overlay_image": ("IMAGE", ),
            "height": (["Full", "Half", "Range"],),
            "side": (["Right", "Left", "Origin"],)
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_checkpoint"

    #OUTPUT_NODE = False

    CATEGORY = "loaders"

    def load_checkpoint(self, base_image, overlay_image, height, side):
        base_image = tensor2pil(base_image).convert('RGBA')
        overlay_image = tensor2pil(overlay_image).convert('RGBA')
        
        base_width, base_height = base_image.size
        if height == 'Full':
            # Resize overlay image to match the height of the base image
            base_width, new_overlay_height = base_image.size
        if height == 'Half':
            # Calculate new height for the overlay image to be between 60% and 40% of the base image height
            new_overlay_height = int(base_height * 0.5)
        if height == 'Range':
            # Calculate new height for the overlay image to be between 60% and 40% of the base image height
            rand_percentage = random.uniform(0.65, 0.75)
            new_overlay_height = int(base_height * rand_percentage)
        
        # Resize overlay image
        overlay_image = overlay_image.resize((int(overlay_image.width * (new_overlay_height / overlay_image.height)), new_overlay_height), Image.ANTIALIAS)
        
        # Calculate position to paste overlay image
        if (side == 'Right'):
            paste_x = 3 * (base_width // 4) - (overlay_image.width // 2)
            paste_y = base_height - overlay_image.height - 25  # Align bottom of overlay image with bottom of base image
        elif (side == 'Left'):
            paste_x = (base_width // 4) - (overlay_image.width // 2)
            paste_y = base_height - overlay_image.height - 25  # Align bottom of overlay image with bottom of base image
        else:
            paste_x = 0
            paste_y = 0

        # Paste overlay image onto base image
        base_image.paste(overlay_image, (paste_x, paste_y), overlay_image)
        
        base_image = pil2tensor(base_image)
        return (base_image,)

class BastardImageOverImageBySize:
    """
    A model loader.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "base_image": ("IMAGE", ),
            "overlay_image": ("IMAGE", ),
            "height": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "side": (["Right", "Left", "Origin"],)
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_checkpoint"

    #OUTPUT_NODE = False

    CATEGORY = "loaders"

    def load_checkpoint(self, base_image, overlay_image, height, side):
        base_image = tensor2pil(base_image).convert('RGBA')
        overlay_image = tensor2pil(overlay_image).convert('RGBA')
        
        base_width, base_height = base_image.size
        new_overlay_height = int(base_height * height)
        
        # Resize overlay image
        overlay_image = overlay_image.resize((int(overlay_image.width * (new_overlay_height / overlay_image.height)), new_overlay_height), Image.ANTIALIAS)
        
        # Calculate position to paste overlay image
        if (side == 'Right'):
            paste_x = 3 * (base_width // 4) - (overlay_image.width // 2)
            paste_y = base_height - overlay_image.height - 25  # Align bottom of overlay image with bottom of base image
        elif (side == 'Left'):
            paste_x = (base_width // 4) - (overlay_image.width // 2)
            paste_y = base_height - overlay_image.height - 25  # Align bottom of overlay image with bottom of base image
        else:
            paste_x = 0
            paste_y = 0

        # Paste overlay image onto base image
        base_image.paste(overlay_image, (paste_x, paste_y), overlay_image)
        
        base_image = pil2tensor(base_image)
        return (base_image,)


class BastardSaveImageAndText:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "images": ("IMAGE", ),
                        "text": ("STRING", {"forceInput": True}),
                        "filepath": ("STRING", {"default": '[time(%Y-%m-%d)]'}),
                        "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                        "filename_delimiter": ("STRING", {"default":"_"}),
                        "filename_number_padding": ("INT", {"default":4, "min":2, "max":9, "step":1}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_image_and_text"
    OUTPUT_NODE = True
    CATEGORY = "Bastard"
    
    def save_image_and_text(self, images, text, filepath, filename_prefix='ComfyUI', filename_delimiter='_', filename_number_padding=4, unique_id=None, prompt=None, extra_pnginfo=None):
    
        tokens = TextTokens()
        filepath = tokens.parseTokens(filepath)
        filename_prefix = tokens.parseTokens(filename_prefix)
        output_path = os.path.join(self.output_dir, filepath)

        if not os.path.exists(output_path):
            print(f"The path `{output_path}` doesn't exist! Creating it...")
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError as e:
                print(f"The path `{output_path}` could not be created! Is there write access?\n{e}")

        if text.strip() == '':
            print(f"There is no text specified to save! Text is empty.")

        delimiter = filename_delimiter
        number_padding = int(filename_number_padding)
        text_extension = '.txt'
        image_extension = '.png'
        text_filename = self.generate_filename(output_path, filename_prefix, delimiter, number_padding, text_extension)
        text_file_path = os.path.join(output_path, text_filename)
        self.writeTextFile(text_file_path, text)
        
        file_basename, _ = os.path.splitext(text_filename)

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(os.path.join(filepath, filename_prefix), self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            if len(images) > 1:
                file = f"{file_basename}{delimiter}{batch_number:02}{image_extension}"
            else:
                file = f"{file_basename}{delimiter}{image_extension}"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results , "text": (text,) } }

    def generate_filename(self, path, prefix, delimiter, number_padding, extension):
        pattern = f"{re.escape(prefix)}{re.escape(delimiter)}(\\d{{{number_padding}}})"
        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(path)
            if re.match(pattern, filename)
        ]
        existing_counters.sort(reverse=True)

        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        filename = f"{prefix}{delimiter}{counter:0{number_padding}}{extension}"
        while os.path.exists(os.path.join(path, filename)):
            counter += 1
            filename = f"{prefix}{delimiter}{counter:0{number_padding}}{extension}"

        return filename

    def writeTextFile(self, file, content):
        try:
            with open(file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
        except OSError:
            print(f"Unable to save file `{file}`")


NODE_CLASS_MAPPINGS = {
    "Bastard 😈: Crop Image By Mask": BastardCropImageByMask,
    "Bastard 😈: Image Over Image":BastardImageOverImage,
    "Bastard 😈: Image Over Image By Size":BastardImageOverImageBySize,
    "Bastard 😈: Isolate Subject By Mask": BastardIsolateSubjectByMask,
    "Bastard 😈: Save Image And Text": BastardSaveImageAndText,
}

# # A dictionary that contains the friendly/humanly readable titles for the nodes
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "BastardCropImageByMask": "BastardCropImageByMask",
#     "BastardImageOverImage":"BastardImageOverImage",
#     "BastardImageOverImageBySize":"BastardImageOverImageBySize",
#     "BastardIsolateSubjectByMask": "BastardIsolateSubjectByMask",
# }
