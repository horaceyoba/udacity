import argparse
def get_input_args():
    # Create Parse using ArgumentParser
    parse=argparse.ArgumentParser(description='Input collection for project')
    
    parse.add_argument('--save_dir', type = str,default='./',help='Checkpoint directory')
    parse.add_argument('--arch',type = str,default='vgg16',help='Type of architecture')
    parse.add_argument('--learning_rate',type = float,default=0.001,help='learning rate of the model')
    parse.add_argument('--hidden_unit1',type = int,default=4096,help='Units of hidden layer1')   
    parse.add_argument('--hidden_unit2',type = int,default=2048,help='Units of hidden layer2')   
    parse.add_argument('--hidden_unit3',type = int,default=1024,help='Units of hidden layer3')   
    parse.add_argument('--epochs',type = int,default=5,help='Number of iterations')    
    parse.add_argument("--gpu", default=argparse.SUPPRESS)
    parse.add_argument('--filename',type = str,default='./flowers/valid/1/image_06739.jpg',help='image for validation')
    parse.add_argument('--top_k',type = str,default=5,help='top k')
    parse.add_argument('--checkpoint',type = str,default='../checkpoint1.pth',help='checkpoint path')
    parse.add_argument('--category_names',type = str,default='cat_to_name.json',help='Names of categories')    
    
    args=parse.parse_args()
    return args