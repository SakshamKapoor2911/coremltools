import os

import torch
import torchvision

import coremltools as ct
from coremltools.converters.mil.frontend.torch.test.testing_utils import TorchBaseTest
from coremltools.converters.mil.frontend.torch.utils import TorchFrontend


class DeformConv2dModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        use_mask,
        use_bias,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_mask = use_mask
        self.weight = torch.nn.Parameter(
            torch.rand(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = torch.nn.Parameter(torch.rand(out_channels)) if use_bias else None

    def forward(self, x, offset, mask=None):
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask if self.use_mask else None,
        )


def output_spatial(input_shape, kernel_size, stride, padding, dilation):
    _, _, h, w = input_shape
    kh, kw = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    h_out = (h + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
    w_out = (w + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1
    return h_out, w_out


CONFIGS = [
    {
        "input_shape": (2, 3, 5, 5),
        "out_channels": 5,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "offset_groups": 1,
        "use_mask": True,
        "use_bias": True,
    },
    {
        "input_shape": (1, 3, 5, 5),
        "out_channels": 4,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "offset_groups": 1,
        "use_mask": False,
        "use_bias": False,
    },
    {
        "input_shape": (1, 2, 4, 4),
        "out_channels": 3,
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "offset_groups": 1,
        "use_mask": True,
        "use_bias": True,
    },
    {
        "input_shape": (1, 4, 5, 5),
        "out_channels": 8,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 2,
        "offset_groups": 1,
        "use_mask": True,
        "use_bias": True,
    },
    {
        "input_shape": (1, 4, 5, 5),
        "out_channels": 6,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "offset_groups": 2,
        "use_mask": True,
        "use_bias": True,
    },
    {
        "input_shape": (1, 4, 5, 5),
        "out_channels": 8,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 2,
        "offset_groups": 2,
        "use_mask": True,
        "use_bias": True,
    },
    {
        "input_shape": (1, 3, 6, 7),
        "out_channels": 4,
        "kernel_size": (2, 3),
        "stride": (2, 1),
        "padding": (1, 2),
        "dilation": (2, 1),
        "groups": 1,
        "offset_groups": 1,
        "use_mask": True,
        "use_bias": True,
    },
]


def main():
    os.environ.setdefault(
        "PYTEST_CURRENT_TEST", "deform_conv2d_fp32_diagnostic::main (call)"
    )
    torch.manual_seed(30)
    for frontend in (TorchFrontend.TORCHSCRIPT, TorchFrontend.TORCHEXPORT):
        for index, config in enumerate(CONFIGS):
            h_out, w_out = output_spatial(
                config["input_shape"],
                config["kernel_size"],
                config["stride"],
                config["padding"],
                config["dilation"],
            )
            offset_shape = (
                config["input_shape"][0],
                2
                * config["offset_groups"]
                * config["kernel_size"][0]
                * config["kernel_size"][1],
                h_out,
                w_out,
            )
            x = torch.randn(config["input_shape"])
            offset = 0.2 * torch.randn(offset_shape)
            input_data = [x, offset]
            if config["use_mask"]:
                mask_shape = (
                    config["input_shape"][0],
                    config["offset_groups"]
                    * config["kernel_size"][0]
                    * config["kernel_size"][1],
                    h_out,
                    w_out,
                )
                input_data.append(torch.rand(mask_shape))

            model = DeformConv2dModel(
                in_channels=config["input_shape"][1],
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                groups=config["groups"],
                use_mask=config["use_mask"],
                use_bias=config["use_bias"],
            ).eval()
            TorchBaseTest.run_compare_torch(
                input_data,
                model,
                input_as_shape=False,
                backend=("mlprogram", "fp32"),
                compute_unit=ct.ComputeUnit.CPU_ONLY,
                atol=1e-4,
                rtol=1e-4,
                frontend=frontend,
            )
            print(f"fp32 native passed frontend={frontend.name} config={index}")


if __name__ == "__main__":
    main()
