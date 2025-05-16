from torchvision import models
import torch.nn as nn
from .base_model import *

class AlexNetFeatureExtractor(BaseFeatureExtractor):
    def build_model(self):
        '''
        Build a fixed-size feature extractor based on AlexNet:
        - Loads pretrained AlexNet from torchvision
        - model.children() contains: [features, avgpool, classifier]
        - Keeps only features and avgpool layers by slicing off the classifier
        - Adds Flatten() to convert [batch_size, 256, 6, 6] → 9216-dim feature vector
          (6×6 comes from AlexNet's architecture when using 224×224 input)
        - The resulting 9216-dimensional vector is consistent and suitable for image retrieval tasks

        '''
            
        model = models.alexnet(pretrained=True)
        # Remove classifier, keep feature extractor
        return nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    def preprocess(self, images, keep_ratio=False,resize=True):
        return default_preprocessing(images, normalise_model = 'imagenet',keep_ratio=keep_ratio)
    
if __name__ == "__main__":
    # Example usage
    extractor = AlexNetFeatureExtractor()
    image_path = "path/to/image.jpg"
    feature = extractor.extract_feature(image_path)
    print("Extracted feature:", feature)

    # Initialize model
    extractor = AlexNetFeatureExtractor()

    # Load database and query
    db_paths = ["images/db1.jpg", "images/db2.jpg", "images/db3.jpg"]
    query_path = "images/query.jpg"

    # Extract features
    db_feats = [extractor.extract_feature(p) for p in db_paths]
    query_feat = extractor.extract_feature(query_path)

    # Retrieve
    topk_indices, topk_scores = compute_similarity(query_feat, db_feats, top_k=3)

    # Output
    print("Top-K matches:", [db_paths[i] for i in topk_indices])
    print("Scores:", topk_scores)

