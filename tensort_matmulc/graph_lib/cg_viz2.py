from graphviz import Digraph
import os

def visualize_tensor_op_graph(
    mlp,
    output,
    file_name="mlp_graph",
    out_dir=None,
    show_shapes=True,
    show_grads=True,
    grad_preview_elems=6
):
    """
    Compatible with the provided MLP class (no changes needed to MLP).
    Reads:
      - mlp.layer_sizes, mlp.weights, mlp.biases
      - mlp._last_logits (for last-layer Z_L grad, if present)
    Infers batch size from `output.shape`.

    Styles:
      - W/B as rectangles
      - tensors (X, WkX, Zk, Outk) as boxes
      - ops (matmul/add/relu/softmax) as ovals
    Layout: top→down (rankdir="TD") to match your current function.
    """

    # --- helpers inside function ---
    def _shape_str(shape_like):
        try:
            return str(tuple(shape_like))
        except Exception:
            return "?"

    def _flatten(x):
        if isinstance(x, (list, tuple)):
            for xi in x:
                yield from _flatten(xi)
        else:
            yield x

    def _fmt_num(x):
        ax = abs(x)
        if ax >= 1e4 or (ax != 0 and ax < 1e-3):
            return f"{x:.2e}"
        if ax < 10:
            return f"{x:.4f}"
        if ax < 1000:
            return f"{x:.2f}"
        return f"{x:.0f}"

    def _preview_list(vals, limit=6):
        vals = list(vals)
        if not vals:
            return ""
        if len(vals) <= limit:
            return "[" + ", ".join(_fmt_num(v) for v in vals) + "]"
        head = ", ".join(_fmt_num(v) for v in vals[:limit])
        return "[" + head + ", …]"

    def _grad_label_line(obj, preview_elems=6):
        # obj can be a TensorT (with .grad and optional .shape/.data) or a raw grad (nested list)
        g = getattr(obj, "grad", None)
        if g is None:
            # maybe obj itself is a grad list
            if isinstance(obj, (list, tuple)):
                flat = list(_flatten(obj))
                prev = _preview_list(flat, limit=preview_elems)
                shp = "?"
                return f"grad {shp}: {prev}" if prev else "grad: None"
            return "grad: None"
        try:
            gshape = getattr(g, "shape", None)
            shp = str(tuple(gshape)) if gshape is not None else "?"
            flat = _flatten(getattr(g, "data", g))
            preview = _preview_list(flat, limit=preview_elems)
            return f"grad {shp}: {preview}"
        except Exception:
            return "grad: <available>"

    def _tensor_label(name, shape_tuple, grad_source=None):
        lines = [name]
        if show_shapes and shape_tuple is not None:
            lines.append(_shape_str(shape_tuple))
        if show_grads:
            lines.append(_grad_label_line(grad_source, grad_preview_elems))
        return "\n".join(lines)
    # --- end helpers ---

    # Infer architecture + batch size
    layer_sizes = getattr(mlp, "layer_sizes", None)
    weights = getattr(mlp, "weights", [])
    biases  = getattr(mlp, "biases", [])
    if layer_sizes is None or not weights or not biases:
        raise ValueError("MLP missing layer_sizes/weights/biases; cannot visualize.")

    # batch size from output (shape = (out_dim, B))
    try:
        B = output.shape[1]
    except Exception:
        B = None  # unknown

    # Graph init
    dot = Digraph("MLPGraph", format="png")
    dot.attr(rankdir="TD", splines="spline", nodesep="0.35", ranksep="0.7")

    param_style  = {"shape": "record", "fontsize": "10"}   # W, B
    tensor_style = {"shape": "box",    "fontsize": "10"}   # X, WkX, Zk, Outk
    op_style     = {"shape": "oval",   "fontsize": "10"}   # matmul/add/relu/softmax

    # Input X
    in_dim = layer_sizes[0]
    x_shape = (in_dim, B) if B is not None else (in_dim, "B")
    # we don't have input grad in this MLP class
    dot.node("X", _tensor_label("X", x_shape, grad_source=None), **tensor_style)
    prev_act_id = "X"

    # Walk layers by definition (no stored intermediates)
    L = len(layer_sizes) - 1  # number of layers
    for i in range(1, L + 1):
        out_dim = layer_sizes[i]
        in_dim  = layer_sizes[i - 1]
        W, b = weights[i - 1], biases[i - 1]

        # ---- parameters ----
        w_id, b_id = f"W{i}", f"B{i}"
        dot.node(w_id, _tensor_label(w_id, (out_dim, in_dim), grad_source=W), **param_style)
        dot.node(b_id, _tensor_label(b_id, (out_dim, 1),     grad_source=B), **param_style)  # B (bias) has grad at `b.grad`

        # ---- matmul op ----
        mm_id = f"MM{i}"
        dot.node(mm_id, f"matmul layer {i}", **op_style)
        dot.edge(prev_act_id, mm_id)  # A_{i-1} -> matmul
        dot.edge(w_id, mm_id)         # W_i -> matmul

        # W_i X tensor
        wx_id = f"W{i}X"
        wx_shape = (out_dim, B) if B is not None else (out_dim, "B")
        dot.node(wx_id, _tensor_label(wx_id, wx_shape, grad_source=None), **tensor_style)
        dot.edge(mm_id, wx_id)

        # ---- add (bias) ----
        add_id = f"ADD{i}"
        dot.node(add_id, f"add layer {i}", **op_style)
        dot.edge(wx_id, add_id)
        dot.edge(b_id, add_id)

        # Z_i (pre-activation)
        z_id = f"Z{i}"
        z_shape = (out_dim, B) if B is not None else (out_dim, "B")

        # For the LAST layer only, if mlp._last_logits exists, use it to show grad
        z_grad_src = None
        if i == L and hasattr(mlp, "_last_logits"):
            z_grad_src = mlp._last_logits  # will show its .grad if set by loss.backward()
        dot.node(z_id, _tensor_label(z_id, z_shape, grad_source=z_grad_src), **tensor_style)
        dot.edge(add_id, z_id)

        # ---- activation ----
        out_id = f"Out{i}"
        out_shape = (out_dim, B) if B is not None else (out_dim, "B")
        is_final = (i == L)

        if not is_final:
            act_id = f"RELU{i}"  # your hidden activation is relu per config
            dot.node(act_id, f"relu layer {i}", **op_style)
            dot.edge(z_id, act_id)
            dot.node(out_id, _tensor_label(out_id, out_shape, grad_source=None), **tensor_style)
            dot.edge(act_id, out_id)
        else:
            # final: softmax (from loss hook), OutL are probabilities
            act_id = f"SOFTMAX{i}"
            dot.node(act_id, f"softmax layer {i}", **op_style)
            dot.edge(z_id, act_id)
            # we can optionally reflect mlp._last_probs, but no grad is stored on it typically
            dot.node(out_id, _tensor_label(out_id, out_shape, grad_source=None), **tensor_style)
            dot.edge(act_id, out_id)

        prev_act_id = out_id

    # Render
    directory = out_dir if out_dir else "."
    os.makedirs(directory, exist_ok=True)
    base = os.path.basename(file_name)
    dot.render(filename=base, directory=directory, cleanup=True)
