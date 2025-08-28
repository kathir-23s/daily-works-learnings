import torch

props = torch.cuda.get_device_properties(0)

print(props)

'''
CudaDeviceProperties(
    name='NVIDIA GeForce RTX 3060', 
    major=8, 
    minor=6, 
    total_memory=11900MB, 
    multi_processor_count=28, 
    uuid=0b42632a-bc5c-f731-724b-6936a19c4b24, 
    pci_bus_id=1, 
    pci_device_id=0, 
    pci_domain_id=0, 
    L2_cache_size=2MB
)
'''