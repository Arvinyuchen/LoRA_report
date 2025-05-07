import torch
import torch.nn as nn
import clip 



def train(model, train_loader, preprocess, device, optimizer, n_epoch, model_path):
    # training mode
    model.train()
    for epoch in range(n_epoch):
        for step, (images, captions) in enumerate(train_loader):

            image_inputs = images.to(device) # [4, 3, 224, 224]
            text_inputs = clip.tokenize(captions).to(device) # [4, 77]

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

            if step % 5 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
    print('Finished Training')
    torch.save(model.state_dict(), model_path)