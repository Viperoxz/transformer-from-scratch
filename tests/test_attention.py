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

def test_multihead_attention():
    batch_size = 2
    seq_len = 3
    d_model = 4
    h = 2
    x = torch.randn(batch_size, seq_len, d_model)
    multihead_attn = MultiHeadedAttention(h, d_model)
    out = multihead_attn(x, x, x)
    print(out)
    assert out.size() == (batch_size, seq_len, d_model)



# test_subsequent_mask()
test_multihead_attention()
print("All tests passed")

# test_subsequent_mask()
