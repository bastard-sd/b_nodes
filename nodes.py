import torch
from PIL import Image, ImageFont, ImageDraw, ImageOps
import numpy as np

# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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

NODE_CLASS_MAPPINGS = {
    "Bastard 😈: Crop Image By Mask": BastardCropImageByMask,
    "Bastard 😈: Image Over Image":BastardImageOverImage,
    "Bastard 😈: Image Over Image By Size":BastardImageOverImageBySize,
    "Bastard 😈: Isolate Subject By Mask": BastardIsolateSubjectByMask,
}

# # A dictionary that contains the friendly/humanly readable titles for the nodes
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "BastardCropImageByMask": "BastardCropImageByMask",
#     "BastardImageOverImage":"BastardImageOverImage",
#     "BastardImageOverImageBySize":"BastardImageOverImageBySize",
#     "BastardIsolateSubjectByMask": "BastardIsolateSubjectByMask",
# }
