import os
import torch
from torch import nn
from Encoders.TextEncoder import createTextEnc, createTokenizer
from Encoders.VisionEncoder import createVisionEncoder
from Encoders.MCP import createMCPEncoder
from Encoders.EncRegistry import createEncRegistry
from Encoders.EncCombine import createEncCombine
from Logic import createRouter
from Experts.ExpRegistry import create_registry
from Experts.TextOutput import createTextOutputExpert
from Experts.Reasoning import createReasoningExpert
from Experts.VisionExpert import createVisionGenExpert

class MultimodalModel(nn.Module):
    def __init__(self, max_steps=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = createTokenizer()

        self.encoder_registry = createEncRegistry()
        self.encoder_registry.add_encoder("text", createTextEnc())
        self.encoder_registry.add_encoder("vision", createVisionEncoder())
        #self.encoder_registry.add_encoder("mcp", createMCPEncoder())

        self.combine = createEncCombine()
        self.router = createRouter()
        self.expert_registry = create_registry()

        self.expert_registry.add_expert("text_output", createTextOutputExpert(max_seq_len=20))
        #self.expert_registry.add_expert("reasoning", createReasoningExpert(num_steps=4))
        #self.expert_registry.add_expert("vision_gen", createVisionGenExpert())

        self.max_steps = max_steps

    def encode(self, input, modality: str, encodings: list = None):
        return self.encoder_registry.encode(input, modality, encodings)
    
    def forward(self, encodings):
        combinedEnc, num_active = self.combine(encodings)
        print(f"Combined shape: {combinedEnc.size()}")

        artifacts = []
        experts_used = []
        for _ in range(self.max_steps):
            routing_vec, state = self.router(combinedEnc)
    
            chosen_expert, scores = self.expert_registry(routing_vec)
            print(f"Expert chosen: {chosen_expert}")

            if chosen_expert == "text_output":
                experts_used.append(chosen_expert)
                expert = self.expert_registry.get_expert(chosen_expert)
                output = expert(state)
                print(f"Text expert output shape: {output.size()}")
                text = self.tokenizer.detokenize(output, strip_special=False)
                return text, artifacts, experts_used
            else:
                experts_used.append(chosen_expert)
                expert = self.expert_registry.get_expert(chosen_expert)
                output, state = expert(state)
                print(f"Next expert output shape: {output.size()}")
                print(f"New state shape: {state.size()}")
                if output is not None:
                    artifacts.append(output)

        return "No response given in time", artifacts, experts_used

    def add_encoder(self, name: str, module: nn.Module,
                    modalities: list[str]):
        self.encoder_registry.add_encoder(name, module, modalities)

    def remove_encoder(self, name: str):
        self.encoder_registry.remove_encoder(name)

    def add_expert(self, name, expert):
        self.expert_registry.add_expert(name, expert)

    def remove_expert(self, name):
        self.expert_registry.remove_expert(name)

    # Component name -> filename mapping
    _COMPONENT_FILES = {
        "encoder_registry": "encoder_registry",
        "combine":          "combiner",
        "router":           "router",
        "expert_registry":  "expert_registry",
    }

    def _get_components(self):
        return {
            "encoder_registry": self.encoder_registry,
            "combine":          self.combine,
            "router":           self.router,
            "expert_registry":  self.expert_registry,
        }

    def store_weights(self, directory):
        os.makedirs(directory, exist_ok=True)

        for name, module in self._get_components().items():
            filename = self._COMPONENT_FILES[name]
            module.store_weights(directory, filename)
            print(f"  Saved {name} → {filename}")

        print(f"All components saved to {directory}/")

    def load_weights(self, directory, strict=True):
        if not os.path.isdir(directory):
            print(f"Directory '{directory}' not found. "
                  f"Starting from scratch.")
            return

        for name, module in self._get_components().items():
            filename = self._COMPONENT_FILES[name]
            filepath = os.path.join(directory, filename)

            if not os.path.isfile(filepath):
                print(f"  {name}: {filename} not found, skipping")
                continue

            try:
                module.load_weights(filepath, strict=True)
                print(f"  {name}: loaded from {filename}")
            except Exception as e:
                print(f"  {name}: failed to load — {e}")

        print(f"Weight loading complete from {directory}/")

model = MultimodalModel()

dummy_image = torch.randn(2, 3, 250, 300)
encodings = model.encode(dummy_image, 'vision')

dummy_video = torch.randn(4, 10, 3, 100, 100) 
encodings = model.encode(dummy_video, 'vision', encodings)

tweet = ["Just set up my new multimodal PyTorch model! 🚀"]
encodings = model.encode(tweet, 'text', encodings)

article = ["PyTorch is an open source machine learning library... " * 100]
encodings = model.encode(article, 'text', encodings)

text, artifacts, experts_used = model.forward(encodings)
print(f"Text output: {text[0]}")
print(f"Number of artifacts created: {len(artifacts)}")
print(f"Sequence of used experts: {experts_used}")