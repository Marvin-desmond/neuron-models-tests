import json 

class ImageNet_Classes:
    def __init__(self) -> None:
        self.classes = None
        with open('./imagenet_classes.json', 'r') as f:
            self.classes = json.load(f) 
        
