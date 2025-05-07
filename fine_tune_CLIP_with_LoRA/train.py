import torch
import torch.nn as nn
import clip 
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np


def train(model, train_loader, preprocess, device, optimizer, n_epoch, model_path):
    # training mode
    model.train()
    for epoch in range(n_epoch):
        for step, (images, captions) in enumerate(train_loader):

            image_inputs = images.to(device)
            text_inputs = clip.tokenize(captions).to(device)

            # forward pass with pre-trained CLIP 
            logits_per_image, logits_per_text = model(image_inputs, text_inputs)
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            # computing function loss
            loss_img = nn.functional.cross_entropy(logits_per_image, ground_truth)
            loss_txt = nn.functional.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_img + loss_txt) / 2

            # updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
    print('Finished Training')
    torch.save(model.state_dict(), model_path)


def eval(model, train, test, device):
    def get_features(dataset):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")