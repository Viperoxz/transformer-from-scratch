# Unit tests for attention
from src.model.utils import subsequent_mask
from src.model.attention import attention, MultiHeadedAttention
import torch

def test_subsequent_mask():
    mask = subsequent_mask(3)
    expected = torch.tensor([[[1, 0, 0],
                               [1, 1, 0],
                               [1, 1, 1]]], dtype=torch.uint8)
    print(mask)
    assert torch.equal(mask, expected) == True 
    print("Test subsequent_mask passed! \n")


def test_multihead_attention():
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    h = 8
    dropout = 0.1

    # Dummy data with explicit sequence length
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # Causal mask
    mask = subsequent_mask(seq_len)

    # Model
    model = MultiHeadedAttention(n_heads=h, d_model=d_model, dropout=dropout)

    # Forward pass
    output = model(query, key, value, mask=mask)

    # Debugging shapes
    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Mask shape:", mask.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", model.attn.shape)

    # Verify
    expected_output_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_output_shape, f"Output shape mismatch! Expected {expected_output_shape}, got {output.shape}"
    print("Test MultiHeadedAttention passed! \n")



test_subsequent_mask()
test_multihead_attention()
print("All tests passed")
