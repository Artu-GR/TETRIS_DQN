import torch
import shutil

def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    #torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, 'checkpoints/best_model.pth.tar')

def load_checkpoint(filename, model, optimizer=None, memory=None):
    #checkpoint = torch.load(filename)
    checkpoint = torch.load(filename, weights_only=False)

    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if memory is not None and 'memory' in checkpoint:
        memory.tree = checkpoint['memory'].tree  # You may need to adapt this
    return checkpoint.get('epoch', 0), checkpoint.get('best_score', None)
