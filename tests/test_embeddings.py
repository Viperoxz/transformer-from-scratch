from src.model.embeddings import Embeddings
import torch

def test_embedding():
    # Parameters
    d_model = 512
    vocab = 1000
    batch_size = 2
    seq_len = 10

    # Model
    model = Embeddings(d_model=d_model, vocab=vocab)

    # Dummy data
    x = torch.randint(low=0, high=vocab, size=(batch_size, seq_len))

    # Forward pass
    output = model(x)

    # Debugging shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Verify
    expected_output_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_output_shape, f"Output shape mismatch! Expected {expected_output_shape}, got {output.shape}"

    print("Test Embeddings passed! \n")

# Run the tests
test_embedding()