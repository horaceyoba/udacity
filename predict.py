from PIL import Image
import numpy as np
from input_args import get_input_args
import torch
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model,class_to_idx

def crop_center(img, w, h):
    img_width, img_height = img.size
    left, right = (img_width - w) / 2, (img_width + w) / 2
    top, bottom = (img_height - h) / 2, (img_height + h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    return img.crop((left, top, right, bottom))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    width, height = im.size
    if width < height:
        im = im.resize((255, int(255*(height/width))))
    else:
        im = im.resize((int(255*(height/width)),255))
    pil_image=crop_center(im,224,224)
    im = np.array(pil_image)
    im = im.transpose((2, 0, 1))
    im = im/255
    im[0] = (im[0] - 0.485)/0.229
    im[1] = (im[1] - 0.456)/0.224
    im[2] = (im[2] - 0.406)/0.225
    im = im[np.newaxis,:]
    image=torch.from_numpy(im)
    image = image.float()
    return image 
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model,class_to_idx, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    inputs=process_image(image_path)
    device = torch.device("cpu")
    if "gpu" in in_arg:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_ps = model.forward(inputs.to(device))
    ps = torch.exp(log_ps)
    probs, classes = ps.topk(topk, dim=1)                    
    reverse_class_idx=dict(map(reversed, class_to_idx.items()))
    class_labels=[reverse_class_idx[k] for k in classes.cpu().numpy()[0]]
    return probs, class_labels

# TODO: Implement the code to predict the class from an image file
def main():
    in_arg = get_input_args()
    model1,class_to_idx = load_checkpoint(in_arg.checkpoint)
    device = torch.device("cpu")
    if "gpu" in in_arg:
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    probs, classes = predict(in_arg.filename, model1,class_to_idx,in_arg.top_k)
    image = process_image(in_arg.filename)
    image1 = image[0,:]
    imshow(image1)
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    class_names=[cat_to_name[k] for k in classes]
    # Get the maximum Probability
    maxElt = torch.max(probs,1)
    maxProb=maxElt[0].item()
    # Get the indices of maximum element in probs array
    indice = maxElt[1].item()
    print("The model is ", maxProb*100, "% certain that the image has a predicted class of ", 
      [class_names[indice] ]  )
    imshow(image1,None,class_names[indice])
    fig, ax = plt.subplots()
    y_pos = np.arange(len(class_names))
    error = np.random.rand(len(class_names))    
    ax.barh(y_pos,probs.tolist()[0], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis() 
    plt.show()

# Call to main function to run the program
if __name__ == "__main__":
    main()