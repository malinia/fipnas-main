import time
import torch

def calculate_inference_time(model, dataset, single_instance=False, device="cpu"):
    model = model.to(device)
    model.eval()
    data_iter = iter(dataset)

    # Measure inference time
    with torch.no_grad():
        start_time = time.perf_counter() 
        for inputs in data_iter:
            inputs = inputs[0].to(device)
            _ = model(inputs)
            if single_instance:
                break
        end_time = time.perf_counter()
    return end_time - start_time


def calculate_gpu_inference_time(model, dataset, single_instance=False, device="cuda"):
    if device != "cuda":
        raise ValueError("CUDA inference requires device to be 'cuda'.")

    model = model.to(device)
    model.eval()
    data_iter = iter(dataset)

    # Measure inference time using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for inputs in data_iter:
            inputs = inputs[0].to(device)
            _ = model(inputs)
            if single_instance:
                break
        end_event.record()

    # Wait for all GPU tasks to complete
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event) / 1000  # Convert from ms to seconds

    return inference_time