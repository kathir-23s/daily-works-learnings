import graphviz
from tensorT_v3 import TensorT  # Adjust the import path according to your project structure

def visualize_tensor_op_graph(mlp, output, file_name='graph'):
    dot = graphviz.Digraph(format='png')
    visited = set()

    total_layers = len(mlp.layers)  # total number of layers

    def _add_nodes(t, layer_num=total_layers):
        tid = f"data_{id(t)}"
        if tid in visited or not (isinstance(t, TensorT) and t.req_grad):
            return
        visited.add(tid)

        data_label = f"shape: {t.shape}\nreq_grad: {t.req_grad}"
        dot.node(tid, data_label, shape='rectangle')

        if t._op:
            opid = f"op_{id(t)}"
            op_name = t._op
            op_label = f"{op_name}\nLayer {layer_num}"
            dot.node(opid, op_label, shape='oval')
            dot.edge(opid, tid)

            for i, parent in enumerate(t._parent):
                if isinstance(parent, TensorT) and parent.req_grad:
                    pid = f"data_{id(parent)}"
                    _add_nodes(parent, layer_num - 1 if op_name == 'relu' else layer_num)

                    edge_label = f'input#{i}'
                    # Skip "Weights(shape)" label if entering activation
                    if i == 0 and len(parent.shape) == 2 and op_name != 'relu':
                        edge_label += f"\nWeights{parent.shape}"
                    edge_label += f"\nLayer {layer_num}"

                    dot.edge(pid, opid, label=edge_label)
        else:
            pass

    _add_nodes(output, layer_num=total_layers)
    dot.render(file_name, view=True)
