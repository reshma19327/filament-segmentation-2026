
import torch
from unetpp_spine import UNetPPSpine


def test_model_output_shape():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        output = model(input_tensor)

    assert output is not None

    if isinstance(output, tuple):
        assert len(output) > 0
        prediction = output[0]
    else:
        prediction = output

    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (1, 1, 256, 256)

def test_model_accepts_batch_of_two_images():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output[0] if isinstance(output, tuple) else output

    assert prediction.shape == (2, 1, 256, 256)

def test_model_output_is_tensor():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output[0] if isinstance(output, tuple) else output

    assert isinstance(prediction, torch.Tensor)

def test_model_output_has_one_channel():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output[0] if isinstance(output, tuple) else output

    assert prediction.shape[1] == 1

def test_model_preserves_spatial_dimensions():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output[0] if isinstance(output, tuple) else output

    assert prediction.shape[-2:] == (256, 256)

def test_model_accepts_128_pixel_image():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(1, 1, 128, 128)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output[0] if isinstance(output, tuple) else output

    assert prediction.shape == (1, 1, 128, 128)


def test_model_accepts_512_pixel_image():
    model = UNetPPSpine()
    model.eval()

    input_tensor = torch.randn(1, 1, 512, 512)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output[0] if isinstance(output, tuple) else output

    assert prediction.shape == (1, 1, 512, 512)
