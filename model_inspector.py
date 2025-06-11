#!/usr/bin/env python3

import torch

try:
    checkpoint = torch.load('model.pth', map_location='cpu')
    print('Model Contents:')

    if isinstance(checkpoint, dict):
        # Handle state dict format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f'Checkpoint keys: {list(checkpoint.keys())}')
        else:
            state_dict = checkpoint

        print('Number of layers:', len(state_dict))
        print('Layer shapes:')
        for name, tensor in state_dict.items():
            print(f'  {name}: {tensor.shape} ({tensor.numel():,} params)')
        print(f'Total params: {sum(t.numel() for t in state_dict.values()):,}')
    else:
        # Handle direct model format
        print('Model type:', type(checkpoint))
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            print('Number of layers:', len(state_dict))
            print('Layer shapes:')
            for name, tensor in state_dict.items():
                print(f'  {name}: {tensor.shape} ({tensor.numel():,} params)')
            print(f'Total params: {sum(t.numel() for t in state_dict.values()):,}')
        else:
            print('Cannot inspect model structure - unknown format')

except FileNotFoundError:
    print("Error: model.pth not found. Please train the model first.")
except Exception as e:
    print(f"Error loading model: {e}")
