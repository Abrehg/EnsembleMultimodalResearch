import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
CONFIG = {
    'input_dim_text': 768,       # Example: BERT/GPT embedding size
    'input_dim_audio': 128,      # Example: Spectrogram feature size
    'input_dim_image': 1024,     # Example: ViT embedding size
    'shared_dim': 512,           # The "Same size output vector" from the diagram
    'num_experts': 4,
    'hidden_dim': 1024
}

# --- 1. The Adaptable Encoding Layer ---
class AdaptableEncoder(nn.Module):
    """
    Projects different modalities into a shared vector space.
    Diagram Note: "All same size output vector"
    """
    def __init__(self, input_dim, shared_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projector(x)

# --- 2. The Experts ---
class Expert(nn.Module):
    """Base class for specific capabilities."""
    def __init__(self, shared_dim, hidden_dim, output_type):
        super().__init__()
        self.output_type = output_type
        self.net = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim) # Returns processed embedding
        )
        
    def forward(self, x):
        # In a real model, this would generate specific modalities.
        # Here we return the processed latent vector and the expert type.
        return self.net(x), self.output_type

# --- 3. The Unified Logic & Router ---
class UnifiedLogicRouter(nn.Module):
    """
    The "Unified Base Logic" + "RecSys-esque" selection method.
    Decides which expert should handle the query.
    """
    def __init__(self, shared_dim, num_experts):
        super().__init__()
        # The "RecSys" is modeled here as a Gating Network
        self.gate = nn.Linear(shared_dim, num_experts)

    def forward(self, x):
        # Calculate logits for each expert
        logits = self.gate(x)
        # Softmax gives us a probability distribution (weights) for the experts
        weights = F.softmax(logits, dim=-1)
        return weights, logits

# --- 4. Main Model Architecture ---
class MultimodalMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Encoders for the left side of the diagram
        self.txt_enc = AdaptableEncoder(config['input_dim_text'], config['shared_dim'])
        self.audio_enc = AdaptableEncoder(config['input_dim_audio'], config['shared_dim'])
        self.img_enc = AdaptableEncoder(config['input_dim_image'], config['shared_dim'])
        # MCP Resources (treated as context vector)
        self.mcp_enc = AdaptableEncoder(config['input_dim_text'], config['shared_dim']) 
        
        self.router = UnifiedLogicRouter(config['shared_dim'], config['num_experts'])
        
        # The "Experts + Logic Loop"
        # 0: Basic Output, 1: Chain of Thought, 2: Audio, 3: Visual
        self.experts = nn.ModuleList([
            Expert(config['shared_dim'], config['hidden_dim'], "Basic Text"),
            Expert(config['shared_dim'], config['hidden_dim'], "Chain of Thought"),
            Expert(config['shared_dim'], config['hidden_dim'], "Audio Artifact"),
            Expert(config['shared_dim'], config['hidden_dim'], "Visual Artifact")
        ])
        
        # Final projection to output (simplified for prototype)
        self.output_head = nn.Linear(config['shared_dim'], config['input_dim_text'])

    def forward(self, inputs):
        """
        inputs: dict containing tensors for available modalities
        """
        embeddings = []
        
        # 1. Adaptable Encoding Layer
        if 'text' in inputs:
            embeddings.append(self.txt_enc(inputs['text']))
        if 'audio' in inputs:
            embeddings.append(self.audio_enc(inputs['audio']))
        if 'image' in inputs:
            embeddings.append(self.img_enc(inputs['image']))
        if 'mcp' in inputs:
            embeddings.append(self.mcp_enc(inputs['mcp']))
            
        if not embeddings:
            return None
            
        # Fuse inputs (Simple average for "Unified Base Logic" representation)
        # Real implementation might use Cross-Attention here.
        unified_vector = torch.stack(embeddings).mean(dim=0)
        
        # 2. RecSys-esque Router Selection
        router_weights, logits = self.router(unified_vector)
        
        # Select the top expert (Hard routing) or mix them (Soft routing).
        # The diagram implies a specific choice ("Output type chosen"), 
        # so we will simulate Hard Routing (picking the best one).
        top_k_weights, top_k_indices = torch.topk(router_weights, 1, dim=-1)
        selected_expert_idx = top_k_indices.item()
        
        # 3. Pass through Selected Expert
        selected_expert = self.experts[selected_expert_idx]
        expert_output, expert_type = selected_expert(unified_vector)
        
        # 4. Final Output Generation
        final_output = self.output_head(expert_output)
        
        return {
            "expert_used": expert_type,
            "router_confidence": top_k_weights.item(),
            "final_vector": final_output,
            "is_artifact": expert_type in ["Audio Artifact", "Visual Artifact"]
        }

# --- Testing the Prototype ---

# Mock Inputs (Random tensors simulating encoded data)
mock_text = torch.randn(1, CONFIG['input_dim_text']) # Batch size 1
mock_img = torch.randn(1, CONFIG['input_dim_image'])

model = MultimodalMoE(CONFIG)

# Example 1: Simulate a text prompt asking for a calculation (Might trigger Logic/CoT)
inputs_1 = {'text': mock_text}
output_1 = model(inputs_1)

# Example 2: Simulate a prompt asking for an image (Text + Visual Context)
inputs_2 = {'text': mock_text, 'image': mock_img}
output_2 = model(inputs_2)

print(f"Scenario 1 (Text Only):")
print(f"  Selected Expert: {output_1['expert_used']}")
print(f"  Confidence: {output_1['router_confidence']:.4f}")
print(f"  Is Artifact: {output_1['is_artifact']}")

print(f"\nScenario 2 (Text + Image):")
print(f"  Selected Expert: {output_2['expert_used']}")
print(f"  Confidence: {output_2['router_confidence']:.4f}")
print(f"  Is Artifact: {output_2['is_artifact']}")