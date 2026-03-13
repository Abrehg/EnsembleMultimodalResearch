from Encoders.TextEncoder import createTextEnc, createTokenizer
from Encoders.VisionEncoder import createVisionEncoder
from Encoders.EncCombine import createEncCombine
from Logic import createRouter
from Experts.SelectExpert import create_registry
from Experts.TextOutput import createTextOutputExpert
import torch
from torch import nn

class MultimodalModel(nn.Module):
    def __init__(self, max_steps=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize encoders
        self.imgEncoder = createVisionEncoder()
        self.tokenizer = createTokenizer()
        self.textEncoder = createTextEnc()
        self.combine = createEncCombine()
        self.router = createRouter()
        self.registry = create_registry()
        plainTextExp = createTextOutputExpert(max_seq_len=20)
        self.registry.add_expert("text_output", plainTextExp)
        self.max_steps = max_steps

    def encodeVision(self, input, encodings:list, modalities:list):
        encoding = self.imgEncoder(input)
        for i in range(encoding.size(0)):
            encodings.append(encoding[i])
            modalities.append('vision')
            print(f"Encoded image output shape: {encoding[i].shape}")
        return encodings, modalities
    
    def encodeText(self, input, encodings:list, modalities:list):
        for i in range(len(input)):
            tokens, _ = self.tokenizer.tokenize_single(input[i])
            encoding = self.textEncoder(tokens)
            encodings.append(encoding)
            modalities.append('text')
            print(f"Encoded text output shape: {encoding.shape}")
        return encodings, modalities
    
    def forward(self, encodings, modalities):
        combinedEnc, num_active = self.combine(encodings, modalities)
        print(f"Combined shape: {combinedEnc.size()}")

        artifacts = []
        experts_used = []
        for _ in range(self.max_steps):
            routing_vec, state = self.router(combinedEnc)
            print(f"New state shape: {state.size()}")
            print(f"Routing vector shape: {routing_vec.size()}")
    
            chosen_expert, scores = self.registry(routing_vec)
            print(f"Expert chosen: {chosen_expert}")
            print(f"Scores: {scores}")

            if chosen_expert == "text_output":
                experts_used.append(chosen_expert)
                expert = self.registry.get_expert(chosen_expert)
                output = expert(state)
                print(f"Text expert output shape: {output.size()}")
                text = self.tokenizer.detokenize(output, strip_special=False)
                return text, artifacts, experts_used
            else:
                experts_used.append(chosen_expert)
                expert = self.registry.get_expert(chosen_expert)
                output, state = expert(state)
                print(f"Next expert output shape: {output.size()}")
                print(f"New state shape: {state.size()}")

        return "No response given in time", artifacts, experts_used

    def add_expert(self, name, expert):
        self.registry.add_expert(name, expert)

    def remove_expert(self, name):
        self.registry.remove_expert(name)

    


model = MultimodalModel()

modalities = []
encodings = []

dummy_image = torch.randn(2, 3, 250, 300)
encodings, modalities = model.encodeVision(dummy_image, encodings, modalities)

dummy_video = torch.randn(4, 10, 3, 100, 100) 
encodings, modalities = model.encodeVision(dummy_video, encodings, modalities)

tweet = ["Just set up my new multimodal PyTorch model! 🚀"]
encodings, modalities = model.encodeText(tweet, encodings, modalities)

article = ["PyTorch is an open source machine learning library... " * 100]
encodings, modalities = model.encodeText(article, encodings, modalities)

text, artifacts, experts_used = model.forward(encodings, modalities)
print(f"Text output: {text[0]}")
print(f"Number of artifacts created: {len(artifacts)}")
print(f"Sequence of used experts: {experts_used}")