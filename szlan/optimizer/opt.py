
class Optimizer:
    
    def __init__(self):
        pass
    
    def ask(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def tell(self, x, y):
        raise NotImplementedError("Subclasses should implement this method.")

    def recommend(self):
        raise NotImplementedError("Subclasses should implement this method.")