import os
import torch
from torch import nn
from Encoders.TextEncoder import createTextEnc
from Encoders.VisionEncoder import createVisionEncoder
from Encoders.LatentState import createLatentStateEncoder
from Encoders.ShortMemory import createShortMemory
from Encoders.EncRegistry import createEncRegistry
from Encoders.EncCombine import createEncCombine
from Logic import createRouter
from Experts.ExpRegistry import create_registry
from Experts.TextOutput import createTextOutputExpert
from Experts.Reasoning import createReasoningExpert
from Experts.LocalMem import createLocalMemExpert
from Experts.InternetSearch import createInternetSearchExpert
from Experts.EOS import createEOSExpert

# Potentially build another encoder that uses previous state in order to build next step in the required task (Ex. for continuing a task)
# Also build a memory system that can be pulled from for input and added to for output. Will require an overall memory module, an encoder that takes in memory module on initialization, and an expert that also takes in memory on initialization
"""
Modality keys
    Encoders:
    - Text input: "text"
    - Vision input: "vision"
    - Latent state matrix: "latent_state"
    - Memory block input: "memory"
    
    Experts:
    - End of Sequence: "eos"
    - Plain text output: "text_output"
    - Reasoning module: "reasoning"
"""
class MultimodalModel(nn.Module):
    def __init__(self, dim=512, max_steps=10, max_memory_entries=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = dim

        self.encoder_registry = createEncRegistry(dim=dim)
        self.encoder_registry.add_encoder("text", createTextEnc(dim=dim))
        self.encoder_registry.add_encoder("vision", createVisionEncoder(dim=dim))
        self.encoder_registry.add_encoder("latent_state", createLatentStateEncoder(dim=dim))
        self.memory = createShortMemory(dim=dim, max_entries=max_memory_entries)
        self.encoder_registry.add_encoder("memory", self.memory)
        
        self.combine = createEncCombine(dim=dim)
        self.router = createRouter(dim=dim)

        self.expert_registry = create_registry(dim=dim)
        self.expert_registry.set_pipeline(
            text_encoder=self.encoder_registry.get_encoder("text"),
            combiner=self.combine,
            router=self.router,
        )
        self.expert_registry.add_expert("eos", createEOSExpert())
        self.expert_registry.add_expert("text_output", createTextOutputExpert(max_seq_len=20, d_model=dim))
        self.expert_registry.add_expert("reasoning", createReasoningExpert(num_steps=4, d_model=dim))

        self.local_mem_expert = createLocalMemExpert(d_model=dim)
        self.expert_registry.add_expert("local_mem", self.local_mem_expert)
        self.local_mem_expert.set_encoders(
            text_encoder=self.encoder_registry.get_encoder("text"),
            vision_encoder=self.encoder_registry.get_encoder("vision"),
        )

        self.internet_search_expert = createInternetSearchExpert(d_model=dim)
        self.expert_registry.add_expert("internet_search", self.internet_search_expert)
        self.internet_search_expert.set_encoders(
            text_encoder=self.encoder_registry.get_encoder("text"),
        )

        self.max_steps = max_steps

    def encode(self, input, modality: str, encodings: list = None):
        return self.encoder_registry.encode(input, modality, encodings)
    
    def forward(self, encodings):
        combinedEnc, _ = self.combine(encodings)
        print(f"Combined shape: {combinedEnc.size()}")

        artifacts = []
        experts_used = []
        for _ in range(self.max_steps):
            routing_vec, state = self.router(combinedEnc)
            chosen_expert, _ = self.expert_registry(routing_vec)
            print(f"Expert chosen: {chosen_expert}")

            if chosen_expert == "eos":
                experts_used.append(chosen_expert)
                return artifacts, state, experts_used
            else:
                experts_used.append(chosen_expert)
                expert = self.expert_registry.get_expert(chosen_expert)
                output, state = expert(state)
                print(f"New state shape: {state.size()}")
                if output is not None:
                    artifacts.append(output)

        return artifacts, state, experts_used

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

# model = MultimodalModel()

# dummy_image = torch.randn(2, 3, 250, 300)
# encodings = model.encode(dummy_image, 'vision')

# dummy_video = torch.randn(4, 10, 3, 100, 100) 
# encodings = model.encode(dummy_video, 'vision', encodings)

# tweet = ["Just set up my new multimodal PyTorch model! 🚀"]
# encodings = model.encode(tweet, 'text', encodings)

# article = ["PyTorch is an open source machine learning library... " * 100]
# encodings = model.encode(article, 'text', encodings)

# artifacts, state, experts_used = model.forward(encodings)
# print(f"Number of artifacts created: {len(artifacts)}")
# print(f"All created artifacts: {artifacts}")
# print(f"Sequence of used experts: {experts_used}")