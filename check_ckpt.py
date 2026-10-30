import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
print('Saved at epoch:', ckpt['epoch'])
print('Val MAE:', round(ckpt['val_mae'], 2))
print('Val RMSE:', round(ckpt['val_rmse'], 2))
print('use_text:', ckpt['use_text'])
