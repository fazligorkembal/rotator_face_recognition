import argparse
from pathlib import Path

import onnx
from onnx import checker, shape_inference


def dim_to_str(dim) -> str:
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    if dim.dim_param:
        return dim.dim_param
    return "?"


def shape_to_str(value_info) -> str:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return "<no-shape>"
    dims = [dim_to_str(dim) for dim in tensor_type.shape.dim]
    return "[" + ", ".join(dims) + "]"


def print_io_shapes(model: onnx.ModelProto, title: str) -> None:
    print(title)
    print("Inputs:")
    for idx, value_info in enumerate(model.graph.input):
        print(f"  [{idx}] {value_info.name}: {shape_to_str(value_info)}")
    print("Outputs:")
    for idx, value_info in enumerate(model.graph.output):
        print(f"  [{idx}] {value_info.name}: {shape_to_str(value_info)}")


def set_batch_dim(value_info, batch_size: int) -> None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape") or len(tensor_type.shape.dim) == 0:
        return

    batch_dim = tensor_type.shape.dim[0]
    batch_dim.ClearField("dim_param")
    batch_dim.dim_value = batch_size


def convert_to_static_batch(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    infer_shapes: bool,
) -> None:
    model = onnx.load(str(input_path))
    print_io_shapes(model, "Original model IO shapes")

    for value_info in model.graph.input:
        set_batch_dim(value_info, batch_size)

    if infer_shapes:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:
            print(f"Shape inference failed, saving model without inferred shapes: {exc}")

    checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    print_io_shapes(model, "Converted model IO shapes")
    print(f"Saved static batch-{batch_size} model to: {output_path}")
    print(
        "Note: If the model contains internal Reshape/Constant nodes hardcoded for batch=1, "
        "those nodes may still need graph surgery."
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "models" / "det_10g.onnx"
    default_output = repo_root / "models" / "det_10g_static_b6.onnx"

    parser = argparse.ArgumentParser(
        description="Rewrite an ONNX model so its input batch dimension becomes a fixed static value."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Source ONNX model path. Default: {default_input}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Destination ONNX model path. Default: {default_output}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Static batch size to write into the ONNX graph. Default: 6",
    )
    parser.add_argument(
        "--skip-shape-inference",
        action="store_true",
        help="Skip ONNX shape inference after rewriting the input batch dimension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_to_static_batch(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        infer_shapes=not args.skip_shape_inference,
    )


if __name__ == "__main__":
    main()
