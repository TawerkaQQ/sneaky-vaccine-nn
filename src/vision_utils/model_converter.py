"""
Convert model to .ptc or .onnx
"""

import typing

import onnx
import onnxslim
import torch
import torchvision


def convert_model_to_torchscript(path_to_weights: str, model: torch.nn.Module):

    checkpoints = torch.load(path_to_weights)

    new_dict = {}
    for key, value in checkpoints["model_state_dict"].items():
        new_dict[key.replace("module.", "")] = value

    model.load_state_dict(new_dict)
    model.eval()

    example_input = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    traced_cpu = torch.jit.trace(model, example_input)
    torch.jit.save(traced_cpu, "torchscript_model.ptc")


def convert_model_to_onnx(
    model: torch.nn.Module,
    path_to_weights: str,
    input_shape: tuple = (1, 3, 224, 224),
    output_path: str = "model.onnx",
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
    use_dynamo: bool = True,
) -> None:
    """
    Args:
        model: PyTorch model
        path_to_weights: path to weights
        input_shape: input_shape (default (1, 3, 224, 224))
        output_path: output path to save model
        input_names: input tensors names
        output_names: output tensors names
        dynamic_axes: dynamic axes
        use_dynamo: use TorchDynamo to export
    """

    checkpoints = torch.load(path_to_weights)

    state_dict_key = "model_state_dict"
    if state_dict_key not in checkpoints:
        state_dict = checkpoints
    else:
        state_dict = checkpoints[state_dict_key]

    new_dict = {}
    for key, value in state_dict.items():
        new_dict[key.replace("module.", "")] = value

    model.load_state_dict(new_dict, strict=False)
    model.eval()

    example_input = torch.randn(input_shape, dtype=torch.float32)

    if input_names is None:
        input_names = [
            f"input_{i}"
            for i in range(len(input_shape) if isinstance(example_input, tuple) else 1)
        ]

    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        example_input,
        output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=use_dynamo,
    )

    model_onnx = onnx.load(output_path)
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, output_path)

    print(f"model save to {output_path}")


if __name__ == "__main__":
    convert_model_to_onnx("./checkpoints/clf_model.pt")
