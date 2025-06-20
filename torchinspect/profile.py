import warnings
from .handlers import handlers
from .utils.trace import trace
import math
__all__ = ['profile_macs']


def profile_macs(model, args=(), kwargs=None, reduction=sum):
    results = dict()

    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node] = func(node)
                break
        else:
            warnings.warn('No handlers found: "{}". Skipped.'.format(
                node.operator))

    if reduction is not None:
        return reduction(results.values())
    else:
        return results

def profile_activations(model, args=(), kwargs=None):
    """Profile the activations of a model."""

    results = dict()

    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        results[node] = 0

        for input in node.inputs:
            if input.shape is not None:
                #results[node] += math.prod(input.shape) 
                print(f"op {node.operator} name {input.name} input: {math.prod(input.shape)}, shape {input.shape} type: {input.dtype}")
        for output in node.outputs:
            if output.shape is not None:
                #results[node] += math.prod(output.shape)
                print(f"op {node.operator} name {output.name} output: {math.prod(output.shape) }, shape {output.shape} type: {output.dtype}")





    return results


def dtype_to_bytes(dtype):
    # Handle common torch dtypes, fallback to 4 bytes (float32)
    if dtype is None:
        return 0
    if hasattr(dtype, 'itemsize'):
        return dtype.itemsize
    # Map known torch dtypes (torch.float32 etc.)
    # You can extend this map as needed
    mapping = {
        'float32': 4,
        'float64': 8,
        'float16': 2,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'int': 4,
        'bool': 1,
        'bfloat16': 2,
        'float': 4,  # Default float type
        'double': 8,  # Default double type
    }
    return mapping.get(str(dtype), 4)  


def profile_activations_data_movement(model, args=(), kwargs=None):
    """Profile activations data movement (in bytes) of a model."""
    
    results = dict()
    graph = trace(model, args, kwargs)  # your graph tracing method
    
    for idx, node in enumerate(graph.nodes):
        data_movement_bytes = 0
        
        for input in node.inputs:
            if input.shape is not None and input.dtype is not None:
                num_elements = math.prod(input.shape)
                bytes_per_element = dtype_to_bytes(input.dtype)
                data_movement_bytes += num_elements # * bytes_per_element
                
        for output in node.outputs:
            if output.shape is not None and output.dtype is not None:
                num_elements = math.prod(output.shape)
                bytes_per_element = dtype_to_bytes(output.dtype)
                data_movement_bytes += num_elements  #* bytes_per_element
        if data_movement_bytes > 0:         
            results[f"{idx}_{node.operator}"] = data_movement_bytes # / 1024**2  # Convert to MB
        

    return results